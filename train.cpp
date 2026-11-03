// ClaudeFish NNUE trainer.  Build: g++ -O3 -march=native -std=c++17 -pthread -o cf_train train.cpp
// Usage:  ./cf_train --data a.txt [--data b.txt ...] --out claudefish.nnue [--epochs 20] [--batch 16384]
//                    [--lr 0.002] [--lambda 0.75] [--threads 4] [--resume net.f32] [--wdl-scale 400]
// Data lines (from "datagen" in the engine):  fen;score_cp_white_pov;result_white_pov
// Net: 768 -> 2*HL (SCReLU) -> 1, 8 material output buckets. Quantisation QA=255, QB=64, scale 400.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
using namespace std;

constexpr int HL = 128, IN = 768, NB = 8, QA = 255, QB = 64;   // keep in sync with the engine (NN_HL)
struct Sample { uint8_t n, stm, bucket; uint8_t pc[32], sq[32]; float target; };

static vector<Sample> loadData(const vector<string>& files, double lambda, double K) {
    vector<Sample> out;
    for (auto& f : files) {
        ifstream in(f);
        string line;
        while (getline(in, line)) {
            size_t a = line.find(';'), b = line.find(';', a + 1);
            if (a == string::npos || b == string::npos) continue;
            string fen = line.substr(0, a);
            double sc = atof(line.substr(a + 1, b - a - 1).c_str()), res = atof(line.c_str() + b + 1);
            Sample s; memset(&s, 0, sizeof(s));
            int r = 7, fl = 0, n = 0; size_t i = 0;
            for (; i < fen.size() && fen[i] != ' '; i++) {
                char ch = fen[i];
                if (ch == '/') { r--; fl = 0; }
                else if (ch >= '1' && ch <= '8') fl += ch - '0';
                else {
                    int c = (ch >= 'A' && ch <= 'Z') ? 0 : 1; int t = -1;
                    switch (tolower(ch)) { case 'p': t = 0; break; case 'n': t = 1; break; case 'b': t = 2; break;
                        case 'r': t = 3; break; case 'q': t = 4; break; case 'k': t = 5; break; }
                    if (t < 0 || n >= 32) { n = -1; break; }
                    s.pc[n] = c * 6 + t; s.sq[n] = r * 8 + fl; n++; fl++;
                }
            }
            if (n < 2) continue;
            s.n = n; s.stm = (i + 1 < fen.size() && fen[i + 1] == 'b') ? 1 : 0;
            s.bucket = min(NB - 1, (n - 2) / 4);
            // white-pov -> side-to-move pov
            double scs = s.stm ? -sc : sc, rs = s.stm ? 1.0 - res : res;
            double p = 1.0 / (1.0 + exp(-scs / K));
            s.target = (float)(lambda * p + (1.0 - lambda) * rs);
            out.push_back(s);
        }
    }
    return out;
}

struct Net {
    vector<float> fw, fb, ow, ob;
    Net() : fw(IN * HL), fb(HL), ow(NB * 2 * HL), ob(NB) {}
    size_t total() const { return fw.size() + fb.size() + ow.size() + ob.size(); }
    float* part(int i) { return i == 0 ? fw.data() : i == 1 ? fb.data() : i == 2 ? ow.data() : ob.data(); }
    size_t psz(int i) const { return i == 0 ? fw.size() : i == 1 ? fb.size() : i == 2 ? ow.size() : ob.size(); }
};
static inline int featIdx(int pcIdx, int sq, int persp) {
    int c = pcIdx / 6, t = pcIdx % 6;
    return ((c == persp ? 0 : 1) * 6 + t) * 64 + (persp == 0 ? sq : (sq ^ 56));
}
static inline float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

// forward (and optionally backward) for one sample; returns squared error
static float run(const Net& net, const Sample& s, Net* g) {
    float x[2][HL];   // [0]=side to move perspective, [1]=other
    int us = s.stm, them = us ^ 1;
    int fi[2][32];
    for (int k = 0; k < 2; k++) memcpy(x[k], net.fb.data(), sizeof(float) * HL);
    for (int i = 0; i < s.n; i++) {
        fi[0][i] = featIdx(s.pc[i], s.sq[i], us); fi[1][i] = featIdx(s.pc[i], s.sq[i], them);
        const float* w0 = &net.fw[fi[0][i] * HL]; const float* w1 = &net.fw[fi[1][i] * HL];
        for (int j = 0; j < HL; j++) { x[0][j] += w0[j]; x[1][j] += w1[j]; }
    }
    const float* ow = &net.ow[s.bucket * 2 * HL];
    float out = net.ob[s.bucket], a[2][HL];
    for (int k = 0; k < 2; k++) for (int j = 0; j < HL; j++) {
        float v = min(max(x[k][j], 0.f), 1.f); a[k][j] = v * v; out += a[k][j] * ow[k * HL + j];
    }
    float p = sigm(out), err = p - s.target;
    if (g) {
        float dout = 2.f * err * p * (1.f - p);
        g->ob[s.bucket] += dout;
        float* gow = &g->ow[s.bucket * 2 * HL];
        float dx[2][HL];
        for (int k = 0; k < 2; k++) for (int j = 0; j < HL; j++) {
            gow[k * HL + j] += dout * a[k][j];
            float v = x[k][j];
            dx[k][j] = (v > 0.f && v < 1.f) ? dout * ow[k * HL + j] * 2.f * v : 0.f;
            g->fb[j] += dx[k][j];
        }
        for (int i = 0; i < s.n; i++) {
            float* g0 = &g->fw[fi[0][i] * HL]; float* g1 = &g->fw[fi[1][i] * HL];
            for (int j = 0; j < HL; j++) { g0[j] += dx[0][j]; g1[j] += dx[1][j]; }
        }
    }
    return err * err;
}

static void writeNet(const Net& net, const string& path) {
    FILE* f = fopen(path.c_str(), "wb");
    fwrite("CFNN", 1, 4, f);
    uint32_t h[6] = {1, HL, IN, NB, QA, QB}; fwrite(h, 4, 6, f);
    auto q16 = [](float v, float scale) { long r = lroundf(v * scale); return (int16_t)max(-32767L, min(32767L, r)); };
    vector<int16_t> t(net.fw.size()); for (size_t i = 0; i < t.size(); i++) t[i] = q16(net.fw[i], QA); fwrite(t.data(), 2, t.size(), f);
    t.assign(net.fb.size(), 0); for (size_t i = 0; i < t.size(); i++) t[i] = q16(net.fb[i], QA); fwrite(t.data(), 2, t.size(), f);
    t.assign(net.ow.size(), 0); for (size_t i = 0; i < t.size(); i++) t[i] = q16(net.ow[i], QB); fwrite(t.data(), 2, t.size(), f);
    for (int b = 0; b < NB; b++) { int32_t v = (int32_t)lroundf(net.ob[b] * QA * QB); fwrite(&v, 4, 1, f); }
    fclose(f);
}

int main(int argc, char** argv) {
    vector<string> data; string outPath = "claudefish.nnue", resume;
    int epochs = 20, batch = 16384, threads = max(1u, thread::hardware_concurrency()); double lr0 = 0.002, lambda = 0.75, K = 400;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto nx = [&]() { return string(argv[++i]); };
        if (a == "--data") data.push_back(nx()); else if (a == "--out") outPath = nx(); else if (a == "--epochs") epochs = atoi(nx().c_str());
        else if (a == "--batch") batch = atoi(nx().c_str()); else if (a == "--lr") lr0 = atof(nx().c_str());
        else if (a == "--lambda") lambda = atof(nx().c_str()); else if (a == "--threads") threads = atoi(nx().c_str());
        else if (a == "--resume") resume = nx(); else if (a == "--wdl-scale") K = atof(nx().c_str());
    }
    if (data.empty()) { fprintf(stderr, "no --data given\n"); return 1; }
    auto samples = loadData(data, lambda, K);
    printf("loaded %zu samples, HL=%d, threads=%d\n", samples.size(), HL, threads); fflush(stdout);
    if (samples.size() < 1000) return 1;
    mt19937_64 rng(42);
    shuffle(samples.begin(), samples.end(), rng);
    size_t nval = samples.size() / 50, ntrain = samples.size() - nval;

    Net net;
    if (!resume.empty()) {
        FILE* f = fopen(resume.c_str(), "rb");
        if (f) { for (int p = 0; p < 4; p++) if (fread(net.part(p), 4, net.psz(p), f) != net.psz(p)) break; fclose(f); printf("resumed from %s\n", resume.c_str()); }
    } else {
        normal_distribution<float> nd(0.f, 1.f);
        for (auto& w : net.fw) w = nd(rng) * 0.06f;
        for (auto& w : net.ow) w = nd(rng) * 0.1f;
    }
    Net m1, m2;   // Adam moments
    for (int p = 0; p < 4; p++) { memset(m1.part(p), 0, 4 * m1.psz(p)); memset(m2.part(p), 0, 4 * m2.psz(p)); }
    vector<Net> grads(threads);
    long step = 0; size_t batchesPerEpoch = (ntrain + batch - 1) / batch; long totalSteps = (long)batchesPerEpoch * epochs;
    vector<size_t> order(ntrain); for (size_t i = 0; i < ntrain; i++) order[i] = i;
    auto t0 = chrono::steady_clock::now();
    for (int ep = 1; ep <= epochs; ep++) {
        shuffle(order.begin(), order.end(), rng);
        double trainLoss = 0;
        for (size_t b0 = 0; b0 < ntrain; b0 += batch) {
            size_t b1 = min(ntrain, b0 + batch);
            vector<double> losses(threads, 0.0);
            vector<thread> th;
            for (int t = 0; t < threads; t++) {
                th.emplace_back([&, t]() {
                    Net& g = grads[t];
                    for (int p = 0; p < 4; p++) memset(g.part(p), 0, 4 * g.psz(p));
                    size_t lo = b0 + (b1 - b0) * t / threads, hi = b0 + (b1 - b0) * (t + 1) / threads;
                    double l = 0;
                    for (size_t i = lo; i < hi; i++) l += run(net, samples[order[i]], &g);
                    losses[t] = l;
                });
            }
            for (auto& x : th) x.join();
            for (double l : losses) trainLoss += l;
            double progress = (double)step / totalSteps;
            double lr = lr0 * (0.01 + 0.99 * 0.5 * (1 + cos(M_PI * progress)));
            step++;
            const float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
            float c1 = 1.f - powf(beta1, (float)step), c2 = 1.f - powf(beta2, (float)step);
            float inv = 1.f / (float)(b1 - b0);
            for (int p = 0; p < 4; p++) {
                float *w = net.part(p), *m = m1.part(p), *v = m2.part(p);
                size_t n = net.psz(p);
                for (size_t i = 0; i < n; i++) {
                    float gr = 0; for (int t = 0; t < threads; t++) gr += grads[t].part(p)[i];
                    gr *= inv;
                    m[i] = beta1 * m[i] + (1 - beta1) * gr; v[i] = beta2 * v[i] + (1 - beta2) * gr * gr;
                    w[i] -= (float)lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps);
                }
            }
            for (int i = 0; i < NB * 2 * HL; i++) net.ow[i] = max(-1.98f, min(1.98f, net.ow[i]));
            for (size_t i = 0; i < net.fw.size(); i++) net.fw[i] = max(-3.f, min(3.f, net.fw[i]));
        }
        double val = 0; for (size_t i = ntrain; i < samples.size(); i++) val += run(net, samples[i], nullptr);
        double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        printf("epoch %2d/%d  train %.5f  val %.5f  lr %.5f  %.0fs\n", ep, epochs, trainLoss / ntrain, val / nval,
               lr0 * (0.01 + 0.99 * 0.5 * (1 + cos(M_PI * (double)step / totalSteps))), secs);
        fflush(stdout);
        writeNet(net, outPath);
        FILE* f = fopen((outPath + ".f32").c_str(), "wb");
        for (int p = 0; p < 4; p++) fwrite(net.part(p), 4, net.psz(p), f);
        fclose(f);
    }
    printf("saved %s\n", outPath.c_str());
    return 0;
}
