// ClaudeFish 3.0 - UCI chess engine written from scratch, now with NNUE.
// Build: g++ -O3 -std=c++17 -pthread -o claudefish claudefish3.cpp   (add -march=native on PC)
//
// NNUE: 768 -> (2 x HL) -> 1 perspective net, SCReLU, 8 material output buckets, int16 quantised,
// incrementally updated accumulators (no king buckets => no refresh needed). Load a net with
// "setoption name EvalFile value <file>" or embed one via cf_net.h. Without a net it falls back to HCE.
//
// Features: bitboards + magic bitboards, legal move generation (pseudo-legal + check test),
// PVS alpha-beta, iterative deepening + aspiration windows, transposition table,
// null-move pruning, late move reductions, futility / reverse futility / razoring,
// late move pruning, SEE, killer + history heuristics, quiescence search,
// tapered hand-crafted evaluation (PeSTO tables + mobility, pawn structure, king safety).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <mutex>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std;
typedef uint64_t U64;
typedef int Move;

enum { WHITE, BLACK };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, NO_PIECE };

constexpr Move NOMOVE = 0;
constexpr int F_CAP = 1 << 15, F_EP = 1 << 16, F_CASTLE = 1 << 17, F_DPUSH = 1 << 18;
constexpr int MAXPLY = 128;
constexpr int MATE = 30000, MATE_BOUND = MATE - 200, INF = 32000, VALUE_NONE = 32001;

// feature switches (UCI spin options 0/1, mainly for A/B testing)
static int OPT_SE = 1;    // singular extensions
static int OPT_CH = 1;    // continuation history + counter moves
static int OPT_TTQ = 1;   // transposition table + check evasions in quiescence
static int OPT_PR = 1;    // extra pruning: SEE for quiets, history pruning
static int OPT_EVX = 1;
static int OPT_PC = 0;    // ProbCut (experimental)
static int OPT_CORR = 0;  // pawn-keyed eval correction history (experimental)
static int OPT_CUT = 0;   // cut-node aware reductions (experimental)   // extra evaluation terms (passed pawns, outposts, mop-up, scaling)

static inline int mFrom(Move m) { return m & 63; }
static inline int mTo(Move m) { return (m >> 6) & 63; }
static inline int mPromo(Move m) { return (m >> 12) & 7; }
static inline Move mk(int f, int t, int promo = 0, int flags = 0) { return f | (t << 6) | (promo << 12) | flags; }

static inline int lsb(U64 b) { return __builtin_ctzll(b); }
static inline int popLsb(U64& b) { int s = __builtin_ctzll(b); b &= b - 1; return s; }
static inline int popcnt(U64 b) { return __builtin_popcountll(b); }
static inline U64 bit(int s) { return 1ULL << s; }
static inline int cheb(int a, int b) { return max(abs((a & 7) - (b & 7)), abs((a >> 3) - (b >> 3))); }
constexpr U64 LIGHT_SQ = 0x55AA55AA55AA55AAULL;

constexpr U64 FILE_A = 0x0101010101010101ULL;
constexpr U64 FILE_H = FILE_A << 7;
constexpr U64 RANK_1 = 0xFFULL;

// ---------------------------------------------------------------- tables
U64 pawnAtt[2][64], knightAtt[64], kingAtt[64];
U64 fileMask[8], adjFiles[8], passedMask[2][64], frontFile[2][64];
U64 zPiece[2][6][64], zCastle[16], zEp[8], zSide;
int cmask[64];

struct RNG {
    U64 s;
    explicit RNG(U64 seed) : s(seed) {}
    U64 next() { s ^= s >> 12; s ^= s << 25; s ^= s >> 27; return s * 2685821657736338717ULL; }
    U64 sparse() { return next() & next() & next(); }
};

struct Magic { U64 mask, magic; U64* att; int shift; };
Magic rookMagic[64], bishopMagic[64];
static U64 rookTbl[0x19000], bishopTbl[0x1480];

static U64 slideAtt(int sq, U64 occ, bool rook) {
    static const int rd[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static const int bd[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    U64 r = 0;
    int rk = sq >> 3, fl = sq & 7;
    for (int d = 0; d < 4; d++) {
        int dr = rook ? rd[d][0] : bd[d][0], df = rook ? rd[d][1] : bd[d][1];
        int nr = rk + dr, nf = fl + df;
        while (nr >= 0 && nr < 8 && nf >= 0 && nf < 8) {
            int s = nr * 8 + nf;
            r |= bit(s);
            if (occ & bit(s)) break;
            nr += dr; nf += df;
        }
    }
    return r;
}

static void initMagics(bool rook) {
    Magic* M = rook ? rookMagic : bishopMagic;
    U64* ptr = rook ? rookTbl : bishopTbl;
    RNG rng(rook ? 0x1234567890ABCDEFULL : 0x0FEDCBA987654321ULL);
    for (int sq = 0; sq < 64; sq++) {
        U64 edges = ((RANK_1 | (RANK_1 << 56)) & ~(RANK_1 << (8 * (sq >> 3)))) |
                    ((FILE_A | FILE_H) & ~(FILE_A << (sq & 7)));
        U64 mask = slideAtt(sq, 0, rook) & ~edges;
        int bits = popcnt(mask);
        int size = 1 << bits;
        M[sq].mask = mask; M[sq].shift = 64 - bits; M[sq].att = ptr; ptr += size;
        vector<U64> occs(size), refs(size);
        U64 b = 0; int n = 0;
        do { occs[n] = b; refs[n] = slideAtt(sq, b, rook); n++; b = (b - mask) & mask; } while (b);
        vector<int> epoch(size, 0);
        int cnt = 0;
        for (;;) {
            U64 magic = rng.sparse();
            if (popcnt((mask * magic) >> 56) < 6) continue;
            cnt++;
            bool ok = true;
            for (int i = 0; i < n; i++) {
                int idx = (int)((occs[i] * magic) >> M[sq].shift);
                if (epoch[idx] < cnt) { epoch[idx] = cnt; M[sq].att[idx] = refs[i]; }
                else if (M[sq].att[idx] != refs[i]) { ok = false; break; }
            }
            if (ok) { M[sq].magic = magic; break; }
        }
    }
}

static inline U64 rookAtt(int sq, U64 occ) { const Magic& m = rookMagic[sq]; return m.att[((occ & m.mask) * m.magic) >> m.shift]; }
static inline U64 bishopAtt(int sq, U64 occ) { const Magic& m = bishopMagic[sq]; return m.att[((occ & m.mask) * m.magic) >> m.shift]; }

static void initTables() {
    for (int sq = 0; sq < 64; sq++) {
        int rk = sq >> 3, fl = sq & 7;
        static const int kn[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
        U64 n = 0, k = 0;
        for (auto& d : kn) { int r = rk + d[0], f = fl + d[1]; if (r >= 0 && r < 8 && f >= 0 && f < 8) n |= bit(r * 8 + f); }
        for (int dr = -1; dr <= 1; dr++) for (int df = -1; df <= 1; df++) {
            if (!dr && !df) continue;
            int r = rk + dr, f = fl + df;
            if (r >= 0 && r < 8 && f >= 0 && f < 8) k |= bit(r * 8 + f);
        }
        knightAtt[sq] = n; kingAtt[sq] = k;
        U64 w = 0, b = 0;
        if (rk < 7) { if (fl > 0) w |= bit(sq + 7); if (fl < 7) w |= bit(sq + 9); }
        if (rk > 0) { if (fl > 0) b |= bit(sq - 9); if (fl < 7) b |= bit(sq - 7); }
        pawnAtt[WHITE][sq] = w; pawnAtt[BLACK][sq] = b;
    }
    for (int f = 0; f < 8; f++) fileMask[f] = FILE_A << f;
    for (int f = 0; f < 8; f++) adjFiles[f] = (f > 0 ? fileMask[f - 1] : 0) | (f < 7 ? fileMask[f + 1] : 0);
    for (int sq = 0; sq < 64; sq++) {
        int rk = sq >> 3, fl = sq & 7;
        U64 aw = rk < 7 ? (~0ULL << (8 * (rk + 1))) : 0;
        U64 ab = rk > 0 ? ((1ULL << (8 * rk)) - 1) : 0;
        frontFile[WHITE][sq] = fileMask[fl] & aw; frontFile[BLACK][sq] = fileMask[fl] & ab;
        passedMask[WHITE][sq] = (fileMask[fl] | adjFiles[fl]) & aw;
        passedMask[BLACK][sq] = (fileMask[fl] | adjFiles[fl]) & ab;
    }
    initMagics(true);
    initMagics(false);
    RNG rng(0xC0FFEE1234567ULL);
    for (int c = 0; c < 2; c++) for (int t = 0; t < 6; t++) for (int s = 0; s < 64; s++) zPiece[c][t][s] = rng.next();
    for (int i = 0; i < 16; i++) zCastle[i] = rng.next();
    for (int i = 0; i < 8; i++) zEp[i] = rng.next();
    zSide = rng.next();
    for (int i = 0; i < 64; i++) cmask[i] = 15;
    cmask[0] = 13; cmask[4] = 12; cmask[7] = 14; cmask[56] = 7; cmask[60] = 3; cmask[63] = 11;
}

// ---------------------------------------------------------------- NNUE data
constexpr int NN_HL = 128, NN_IN = 768, NN_BUCKETS = 8, NN_QA = 255, NN_QB = 64, NN_SCALE = 400;
struct alignas(64) NNData {
    int16_t fw[NN_IN * NN_HL];
    int16_t fb[NN_HL];
    int16_t ow[NN_BUCKETS][2 * NN_HL];
    int32_t ob[NN_BUCKETS];
    bool loaded;
};
static NNData NN;
struct alignas(32) Acc { int16_t v[2][NN_HL]; };
static bool g_useNNUE = false;   // default off until a net that beats the HCE is installed
static string g_netName = "none";

// file layout: "CFNN", u32 version(1), u32 HL, u32 IN, u32 buckets, u32 QA, u32 QB, then
// int16 fw[IN*HL], int16 fb[HL], int16 ow[buckets][2*HL], int32 ob[buckets]  (little endian)
static size_t nnPayloadBytes() { return sizeof(NN.fw) + sizeof(NN.fb) + sizeof(NN.ow) + sizeof(NN.ob); }
static bool nnLoadBuf(const unsigned char* buf, size_t len, const string& name) {
    const size_t hdr = 4 + 6 * 4;
    if (len != hdr + nnPayloadBytes() || memcmp(buf, "CFNN", 4) != 0) return false;
    uint32_t h[6]; memcpy(h, buf + 4, sizeof(h));
    if (h[0] != 1 || h[1] != NN_HL || h[2] != NN_IN || h[3] != NN_BUCKETS || h[4] != NN_QA || h[5] != NN_QB) return false;
    const unsigned char* p = buf + hdr;
    memcpy(NN.fw, p, sizeof(NN.fw)); p += sizeof(NN.fw);
    memcpy(NN.fb, p, sizeof(NN.fb)); p += sizeof(NN.fb);
    memcpy(NN.ow, p, sizeof(NN.ow)); p += sizeof(NN.ow);
    memcpy(NN.ob, p, sizeof(NN.ob));
    NN.loaded = true; g_netName = name;
    return true;
}
static bool nnLoadFile(const string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return false; }
    vector<unsigned char> buf((size_t)sz);
    bool ok = fread(buf.data(), 1, buf.size(), f) == buf.size();
    fclose(f);
    if (!ok) return false;
    size_t slash = path.find_last_of("/\\");
    return nnLoadBuf(buf.data(), buf.size(), slash == string::npos ? path : path.substr(slash + 1));
}
#if defined(__has_include)
#if __has_include("cf_net.h")
#include "cf_net.h"
#define CF_HAVE_EMBEDDED 1
#endif
#endif

// ---------------------------------------------------------------- position
struct State { U64 key; int castle, ep, half, captured; Move mv; };

struct Pos {
    U64 bb[2][6], occ[2], all, key;
    int pt[64], stm, castle, ep, half, hp;
    State hist[2048 + MAXPLY + 8];
    Acc acc[MAXPLY + 8];   // NNUE accumulator stack (only maintained when track == true)
    int ai = 0;
    bool track = false;

    // accumulator helpers: feature index = ((own?0:1)*6 + type)*64 + (square seen from the perspective)
    inline void accAdd(int c, int t, int sq) {
        Acc& a = acc[ai];
        for (int persp = 0; persp < 2; persp++) {
            int idx = ((c == persp ? 0 : 1) * 6 + t) * 64 + (persp == WHITE ? sq : (sq ^ 56));
            const int16_t* w = NN.fw + idx * NN_HL;
            int16_t* v = a.v[persp];
            for (int j = 0; j < NN_HL; j++) v[j] += w[j];
        }
    }
    inline void accSub(int c, int t, int sq) {
        Acc& a = acc[ai];
        for (int persp = 0; persp < 2; persp++) {
            int idx = ((c == persp ? 0 : 1) * 6 + t) * 64 + (persp == WHITE ? sq : (sq ^ 56));
            const int16_t* w = NN.fw + idx * NN_HL;
            int16_t* v = a.v[persp];
            for (int j = 0; j < NN_HL; j++) v[j] -= w[j];
        }
    }
    void refreshAcc() {
        ai = 0;
        for (int persp = 0; persp < 2; persp++) memcpy(acc[0].v[persp], NN.fb, sizeof(NN.fb));
        for (int c = 0; c < 2; c++) for (int t = 0; t < 6; t++) {
            U64 b = bb[c][t];
            while (b) accAdd(c, t, popLsb(b));
        }
    }
    string fen() const {
        string s;
        for (int r = 7; r >= 0; r--) {
            int e = 0;
            for (int f = 0; f < 8; f++) {
                int sq = r * 8 + f, t = pt[sq];
                if (t == NO_PIECE) { e++; continue; }
                if (e) { s += char('0' + e); e = 0; }
                char ch = "pnbrqk"[t];
                s += (occ[WHITE] & bit(sq)) ? char(toupper(ch)) : ch;
            }
            if (e) s += char('0' + e);
            if (r) s += '/';
        }
        s += stm == WHITE ? " w " : " b ";
        string c;
        if (castle & 1) c += 'K'; if (castle & 2) c += 'Q'; if (castle & 4) c += 'k'; if (castle & 8) c += 'q';
        s += c.empty() ? "-" : c;
        s += ' ';
        if (ep == -1) s += '-'; else { s += char('a' + (ep & 7)); s += char('1' + (ep >> 3)); }
        s += ' ' + to_string(half) + " 1";
        return s;
    }

    void clear() {
        memset(bb, 0, sizeof(bb)); occ[0] = occ[1] = all = key = 0;
        for (int i = 0; i < 64; i++) pt[i] = NO_PIECE;
        stm = WHITE; castle = 0; ep = -1; half = 0; hp = 0;
    }
    void put(int c, int t, int sq) {
        bb[c][t] |= bit(sq); occ[c] |= bit(sq); all |= bit(sq); pt[sq] = t; key ^= zPiece[c][t][sq];
        if (track) accAdd(c, t, sq);
    }
    void remove(int c, int t, int sq) {
        bb[c][t] &= ~bit(sq); occ[c] &= ~bit(sq); all &= ~bit(sq); pt[sq] = NO_PIECE; key ^= zPiece[c][t][sq];
        if (track) accSub(c, t, sq);
    }
    bool setFen(const string& fen) {
        clear();
        istringstream ss(fen);
        string board, side, cas, eps;
        if (!(ss >> board >> side)) return false;
        if (!(ss >> cas)) cas = "-";
        if (!(ss >> eps)) eps = "-";
        int h = 0; ss >> h; half = h;
        int r = 7, f = 0;
        for (char ch : board) {
            if (ch == '/') { r--; f = 0; }
            else if (isdigit((unsigned char)ch)) f += ch - '0';
            else {
                int c = isupper((unsigned char)ch) ? WHITE : BLACK;
                int t = NO_PIECE;
                switch (tolower(ch)) { case 'p': t = PAWN; break; case 'n': t = KNIGHT; break; case 'b': t = BISHOP; break;
                    case 'r': t = ROOK; break; case 'q': t = QUEEN; break; case 'k': t = KING; break; }
                if (t == NO_PIECE || r < 0 || f > 7) return false;
                put(c, t, r * 8 + f); f++;
            }
        }
        stm = side == "b" ? BLACK : WHITE;
        for (char ch : cas) { if (ch == 'K') castle |= 1; if (ch == 'Q') castle |= 2; if (ch == 'k') castle |= 4; if (ch == 'q') castle |= 8; }
        ep = -1;
        if (eps.size() == 2 && eps[0] >= 'a' && eps[0] <= 'h') {
            int sq = (eps[1] - '1') * 8 + (eps[0] - 'a');
            int us = stm;  // side to move captures
            if (pawnAtt[us ^ 1][sq] & bb[us][PAWN]) ep = sq;
        }
        key ^= zCastle[castle];
        if (ep != -1) key ^= zEp[ep & 7];
        if (stm == BLACK) key ^= zSide;
        if (!bb[0][KING] || !bb[1][KING]) return false;
        return true;
    }

    U64 attackersTo(int sq, U64 o) const {
        return (pawnAtt[BLACK][sq] & bb[WHITE][PAWN]) | (pawnAtt[WHITE][sq] & bb[BLACK][PAWN]) |
               (knightAtt[sq] & (bb[0][KNIGHT] | bb[1][KNIGHT])) | (kingAtt[sq] & (bb[0][KING] | bb[1][KING])) |
               (bishopAtt(sq, o) & (bb[0][BISHOP] | bb[1][BISHOP] | bb[0][QUEEN] | bb[1][QUEEN])) |
               (rookAtt(sq, o) & (bb[0][ROOK] | bb[1][ROOK] | bb[0][QUEEN] | bb[1][QUEEN]));
    }
    bool isAttacked(int sq, int by) const {
        return (pawnAtt[by ^ 1][sq] & bb[by][PAWN]) || (knightAtt[sq] & bb[by][KNIGHT]) || (kingAtt[sq] & bb[by][KING]) ||
               (bishopAtt(sq, all) & (bb[by][BISHOP] | bb[by][QUEEN])) || (rookAtt(sq, all) & (bb[by][ROOK] | bb[by][QUEEN]));
    }
    bool inCheck() const { return isAttacked(lsb(bb[stm][KING]), stm ^ 1); }
    bool illegalAfterMake() const { return isAttacked(lsb(bb[stm ^ 1][KING]), stm); }

    void make(Move m) {
        State& s = hist[hp++];
        s.key = key; s.castle = castle; s.ep = ep; s.half = half; s.mv = m;
        if (track) { acc[ai + 1] = acc[ai]; ai++; }
        int from = mFrom(m), to = mTo(m), promo = mPromo(m);
        int us = stm, them = us ^ 1;
        int type = pt[from];
        int captured = NO_PIECE;
        if (ep != -1) { key ^= zEp[ep & 7]; ep = -1; }
        half++;
        if (m & F_EP) { int cs = to + (us == WHITE ? -8 : 8); captured = PAWN; remove(them, PAWN, cs); }
        else if (m & F_CAP) { captured = pt[to]; remove(them, captured, to); }
        s.captured = captured;
        remove(us, type, from);
        put(us, promo ? promo : type, to);
        if (m & F_CASTLE) {
            if (to == 6) { remove(us, ROOK, 7); put(us, ROOK, 5); }
            else if (to == 2) { remove(us, ROOK, 0); put(us, ROOK, 3); }
            else if (to == 62) { remove(us, ROOK, 63); put(us, ROOK, 61); }
            else { remove(us, ROOK, 56); put(us, ROOK, 59); }
        }
        if (type == PAWN) {
            half = 0;
            if (m & F_DPUSH) {
                int mid = (from + to) / 2;
                if (pawnAtt[us][mid] & bb[them][PAWN]) { ep = mid; key ^= zEp[ep & 7]; }
            }
        }
        if (captured != NO_PIECE) half = 0;
        key ^= zCastle[castle];
        castle &= cmask[from] & cmask[to];
        key ^= zCastle[castle];
        stm ^= 1; key ^= zSide;
    }
    void unmake() {
        State& s = hist[--hp];
        Move m = s.mv;
        stm ^= 1;
        bool trk = track; track = false;   // the accumulator is simply popped, no incremental work
        int us = stm, them = us ^ 1;
        int from = mFrom(m), to = mTo(m), promo = mPromo(m);
        int movedType = promo ? PAWN : pt[to];
        remove(us, pt[to], to);
        put(us, movedType, from);
        if (m & F_EP) put(them, PAWN, to + (us == WHITE ? -8 : 8));
        else if (s.captured != NO_PIECE) put(them, s.captured, to);
        if (m & F_CASTLE) {
            if (to == 6) { remove(us, ROOK, 5); put(us, ROOK, 7); }
            else if (to == 2) { remove(us, ROOK, 3); put(us, ROOK, 0); }
            else if (to == 62) { remove(us, ROOK, 61); put(us, ROOK, 63); }
            else { remove(us, ROOK, 59); put(us, ROOK, 56); }
        }
        key = s.key; castle = s.castle; ep = s.ep; half = s.half;
        track = trk; if (trk) ai--;
    }
    void makeNull() {
        State& s = hist[hp++];
        s.key = key; s.castle = castle; s.ep = ep; s.half = half; s.mv = NOMOVE; s.captured = NO_PIECE;
        if (ep != -1) { key ^= zEp[ep & 7]; ep = -1; }
        stm ^= 1; key ^= zSide; half = 0;
    }
    void unmakeNull() {
        State& s = hist[--hp];
        stm ^= 1; key = s.key; castle = s.castle; ep = s.ep; half = s.half;
    }
    bool repetition() const {
        int n = min(half, hp);
        for (int i = 2; i <= n; i += 2) if (hist[hp - i].key == key) return true;
        return false;
    }

    int gen(Move* mv, bool caps) const {
        int n = 0, us = stm, them = us ^ 1;
        U64 own = occ[us], enemy = occ[them];
        U64 pawns = bb[us][PAWN];
        int up = us == WHITE ? 8 : -8;
        int promoRank = us == WHITE ? 6 : 1, startRank = us == WHITE ? 1 : 6;
        while (pawns) {
            int from = popLsb(pawns), to = from + up, r = from >> 3;
            bool prom = (r == promoRank);
            if (!(all & bit(to))) {
                if (prom) {
                    mv[n++] = mk(from, to, QUEEN);
                    if (!caps) { mv[n++] = mk(from, to, ROOK); mv[n++] = mk(from, to, BISHOP); mv[n++] = mk(from, to, KNIGHT); }
                } else if (!caps) {
                    mv[n++] = mk(from, to);
                    if (r == startRank && !(all & bit(to + up))) mv[n++] = mk(from, to + up, 0, F_DPUSH);
                }
            }
            U64 a = pawnAtt[us][from] & enemy;
            while (a) {
                int t = popLsb(a);
                if (prom) { mv[n++] = mk(from, t, QUEEN, F_CAP); mv[n++] = mk(from, t, ROOK, F_CAP);
                            mv[n++] = mk(from, t, BISHOP, F_CAP); mv[n++] = mk(from, t, KNIGHT, F_CAP); }
                else mv[n++] = mk(from, t, 0, F_CAP);
            }
            if (ep != -1 && (pawnAtt[us][from] & bit(ep))) mv[n++] = mk(from, ep, 0, F_CAP | F_EP);
        }
        for (int t = KNIGHT; t <= KING; t++) {
            U64 b = bb[us][t];
            while (b) {
                int from = popLsb(b);
                U64 a;
                switch (t) {
                    case KNIGHT: a = knightAtt[from]; break;
                    case BISHOP: a = bishopAtt(from, all); break;
                    case ROOK: a = rookAtt(from, all); break;
                    case QUEEN: a = bishopAtt(from, all) | rookAtt(from, all); break;
                    default: a = kingAtt[from]; break;
                }
                a &= caps ? enemy : ~own;
                while (a) { int to = popLsb(a); mv[n++] = mk(from, to, 0, (enemy & bit(to)) ? F_CAP : 0); }
            }
        }
        if (!caps) {
            if (us == WHITE) {
                if ((castle & 1) && !(all & (bit(5) | bit(6))) && !isAttacked(4, BLACK) && !isAttacked(5, BLACK) && !isAttacked(6, BLACK))
                    mv[n++] = mk(4, 6, 0, F_CASTLE);
                if ((castle & 2) && !(all & (bit(1) | bit(2) | bit(3))) && !isAttacked(4, BLACK) && !isAttacked(3, BLACK) && !isAttacked(2, BLACK))
                    mv[n++] = mk(4, 2, 0, F_CASTLE);
            } else {
                if ((castle & 4) && !(all & (bit(61) | bit(62))) && !isAttacked(60, WHITE) && !isAttacked(61, WHITE) && !isAttacked(62, WHITE))
                    mv[n++] = mk(60, 62, 0, F_CASTLE);
                if ((castle & 8) && !(all & (bit(57) | bit(58) | bit(59))) && !isAttacked(60, WHITE) && !isAttacked(59, WHITE) && !isAttacked(58, WHITE))
                    mv[n++] = mk(60, 58, 0, F_CASTLE);
            }
        }
        return n;
    }

    // static exchange evaluation (swap algorithm)
    int see(Move m) const {
        static const int V[6] = {100, 320, 330, 500, 900, 20000};
        int from = mFrom(m), to = mTo(m), us = stm;
        int gain[40], d = 0;
        int victim = (m & F_EP) ? PAWN : pt[to];
        gain[0] = victim == NO_PIECE ? 0 : V[victim];
        int attacker = pt[from];
        U64 o = all ^ bit(from);
        if (m & F_EP) o ^= bit(to + (us == WHITE ? -8 : 8));
        U64 att = attackersTo(to, o) & o;
        int side = us ^ 1;
        for (;;) {
            d++;
            gain[d] = V[attacker] - gain[d - 1];
            if (max(-gain[d - 1], gain[d]) < 0) break;
            U64 cand = 0; int t;
            for (t = PAWN; t <= KING; t++) { cand = att & bb[side][t]; if (cand) break; }
            if (t > KING) break;
            o ^= cand & (~cand + 1);
            att = attackersTo(to, o) & o;
            attacker = t; side ^= 1;
            if (d >= 38) break;
        }
        while (--d) gain[d - 1] = -max(-gain[d - 1], gain[d]);
        return gain[0];
    }
};

static const char* START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static string moveStr(Move m) {
    string s;
    s += char('a' + (mFrom(m) & 7)); s += char('1' + (mFrom(m) >> 3));
    s += char('a' + (mTo(m) & 7)); s += char('1' + (mTo(m) >> 3));
    if (mPromo(m)) s += "nbrq"[mPromo(m) - 1];
    return s;
}
static int legalMoves(Pos& p, Move* out) {
    Move tmp[256];
    int n = p.gen(tmp, false), k = 0;
    for (int i = 0; i < n; i++) {
        p.make(tmp[i]);
        if (!p.illegalAfterMake()) out[k++] = tmp[i];
        p.unmake();
    }
    return k;
}
static Move parseMove(Pos& p, const string& s) {
    Move mv[256];
    int n = legalMoves(p, mv);
    for (int i = 0; i < n; i++) if (moveStr(mv[i]) == s) return mv[i];
    return NOMOVE;
}
static U64 perft(Pos& p, int d) {
    if (d == 0) return 1;
    Move mv[256];
    int n = p.gen(mv, false);
    U64 total = 0;
    for (int i = 0; i < n; i++) {
        p.make(mv[i]);
        if (!p.illegalAfterMake()) total += (d == 1) ? 1 : perft(p, d - 1);
        p.unmake();
    }
    return total;
}

// ---------------------------------------------------------------- evaluation
static const int MGV[6] = {82, 337, 365, 477, 1025, 0};
static const int EGV[6] = {94, 281, 297, 512, 936, 0};
static const int PHW[6] = {0, 1, 1, 2, 4, 0};

// PeSTO piece-square tables, written visually (rank 8 first). White index = sq ^ 56, black index = sq.
static const int PST_MG[6][64] = {
 {0,0,0,0,0,0,0,0, 98,134,61,95,68,126,34,-11, -6,7,26,31,65,56,25,-20, -14,13,6,21,23,12,17,-23,
  -27,-2,-5,12,17,6,10,-25, -26,-4,-4,-10,3,3,33,-12, -35,-1,-20,-23,-15,24,38,-22, 0,0,0,0,0,0,0,0},
 {-167,-89,-34,-49,61,-97,-15,-107, -73,-41,72,36,23,62,7,-17, -47,60,37,65,84,129,73,44, -9,17,19,53,37,69,18,22,
  -13,4,16,13,28,19,21,-8, -23,-9,12,10,19,17,25,-16, -29,-53,-12,-3,-1,18,-14,-19, -105,-21,-58,-33,-17,-28,-19,-23},
 {-29,4,-82,-37,-25,-42,7,-8, -26,16,-18,-13,30,59,18,-47, -16,37,43,40,35,50,37,-2, -4,5,19,50,37,37,7,-2,
  -6,13,13,26,34,12,10,4, 0,15,15,15,14,27,18,10, 4,15,16,0,7,21,33,1, -33,-3,-14,-21,-13,-12,-39,-21},
 {32,42,32,51,63,9,31,43, 27,32,58,62,80,67,26,44, -5,19,26,36,17,45,61,16, -24,-11,7,26,24,35,-8,-20,
  -36,-26,-12,-1,9,-7,6,-23, -45,-25,-16,-17,3,0,-5,-33, -44,-16,-20,-9,-1,11,-6,-71, -19,-13,1,17,16,7,-37,-26},
 {-28,0,29,12,59,44,43,45, -24,-39,-5,1,-16,57,28,54, -13,-17,7,8,29,56,47,57, -27,-27,-16,-16,-1,17,-2,1,
  -9,-26,-9,-10,-2,-4,3,-3, -14,2,-11,-2,-5,2,14,5, -35,-8,11,2,8,15,-3,1, -1,-18,-9,10,-15,-25,-31,-50},
 {-65,23,16,-15,-56,-34,2,13, 29,-1,-20,-7,-8,-4,-38,-29, -9,24,2,-16,-20,6,22,-22, -17,-20,-12,-27,-30,-25,-14,-36,
  -49,-1,-27,-39,-46,-44,-33,-51, -14,-14,-22,-46,-44,-30,-15,-27, 1,7,-8,-64,-43,-16,9,8, -15,36,12,-54,8,-28,24,14}};
static const int PST_EG[6][64] = {
 {0,0,0,0,0,0,0,0, 178,173,158,134,147,132,165,187, 94,100,85,67,56,53,82,84, 32,24,13,5,-2,4,17,17,
  13,9,-3,-7,-7,-8,3,-1, 4,7,-6,1,0,-5,-1,-8, 13,8,8,10,13,0,2,-7, 0,0,0,0,0,0,0,0},
 {-58,-38,-13,-28,-31,-27,-63,-99, -25,-8,-25,-2,-9,-25,-24,-52, -24,-20,10,9,-1,-9,-19,-41, -17,3,22,22,22,11,8,-18,
  -18,-6,16,25,16,17,4,-18, -23,-3,-1,15,10,-3,-20,-22, -42,-20,-10,-5,-2,-20,-23,-44, -29,-51,-23,-15,-22,-18,-50,-64},
 {-14,-21,-11,-8,-7,-9,-17,-24, -8,-4,7,-12,-3,-13,-4,-14, 2,-8,0,-1,-2,6,0,4, -3,9,12,9,14,10,3,2,
  -6,3,13,19,7,10,-3,-9, -12,-3,8,10,13,3,-7,-15, -14,-18,-7,-1,4,-9,-15,-27, -23,-9,-23,-5,-9,-16,-5,-17},
 {13,10,18,15,12,12,8,5, 11,13,13,11,-3,3,8,3, 7,7,7,5,4,-3,-5,-3, 4,3,13,1,2,1,-1,2,
  3,5,8,4,-5,-6,-8,-11, -4,0,-5,-1,-7,-12,-8,-16, -6,-6,0,2,-9,-9,-11,-3, -9,2,3,-1,-5,-13,4,-20},
 {-9,22,22,27,27,19,10,20, -17,20,32,41,58,25,30,0, -20,6,9,49,47,35,19,9, 3,22,24,45,57,40,57,36,
  -18,28,19,47,31,34,39,23, -16,-27,15,6,9,17,10,5, -22,-23,-30,-16,-16,-23,-36,-32, -33,-28,-22,-43,-5,-32,-20,-41},
 {-74,-35,-18,-18,-11,15,4,-17, -12,17,14,17,17,38,23,11, 10,17,23,15,20,45,44,13, -8,22,24,27,26,33,26,3,
  -18,-4,21,24,27,23,9,-11, -19,-3,11,21,23,16,7,-9, -27,-11,4,13,14,4,-5,-17, -53,-34,-21,-11,-28,-14,-24,-43}};

static int mgT[2][6][64], egT[2][6][64];
static const int PASSED_MG[8] = {0, 2, 5, 12, 28, 55, 95, 0};
static const int PASSED_EG[8] = {0, 5, 12, 28, 55, 100, 160, 0};
static const int ATK_W[6] = {0, 2, 2, 3, 5, 0};

static void initEval() {
    for (int t = 0; t < 6; t++) for (int s = 0; s < 64; s++) {
        mgT[WHITE][t][s] = MGV[t] + PST_MG[t][s ^ 56]; egT[WHITE][t][s] = EGV[t] + PST_EG[t][s ^ 56];
        mgT[BLACK][t][s] = MGV[t] + PST_MG[t][s];      egT[BLACK][t][s] = EGV[t] + PST_EG[t][s];
    }
}

static inline U64 wPawnAttacks(U64 p) { return ((p << 7) & ~FILE_H) | ((p << 9) & ~FILE_A); }
static inline U64 bPawnAttacks(U64 p) { return ((p >> 7) & ~FILE_A) | ((p >> 9) & ~FILE_H); }

static int evaluateHCE(const Pos& p) {
    int mg[2] = {0, 0}, eg[2] = {0, 0}, phase = 0;
    if (!p.bb[0][PAWN] && !p.bb[1][PAWN] && !(p.bb[0][ROOK] | p.bb[1][ROOK] | p.bb[0][QUEEN] | p.bb[1][QUEEN])) {
        int minors = popcnt(p.bb[0][KNIGHT] | p.bb[0][BISHOP] | p.bb[1][KNIGHT] | p.bb[1][BISHOP]);
        if (minors <= 1) return 0;
    }
    U64 pawnAttBB[2] = {wPawnAttacks(p.bb[0][PAWN]), bPawnAttacks(p.bb[1][PAWN])};
    int ksq[2] = {lsb(p.bb[0][KING]), lsb(p.bb[1][KING])};
    U64 kzone[2] = {kingAtt[ksq[0]] | bit(ksq[0]), kingAtt[ksq[1]] | bit(ksq[1])};
    int atkUnits[2] = {0, 0}, atkCnt[2] = {0, 0};

    for (int c = 0; c < 2; c++) {
        int them = c ^ 1;
        U64 own = p.occ[c], ownP = p.bb[c][PAWN], enP = p.bb[them][PAWN];
        for (int t = PAWN; t <= KING; t++) {
            U64 b = p.bb[c][t];
            while (b) {
                int sq = popLsb(b);
                mg[c] += mgT[c][t][sq]; eg[c] += egT[c][t][sq]; phase += PHW[t];
                U64 a = 0;
                switch (t) {
                    case PAWN: {
                        int f = sq & 7, rr = c == WHITE ? (sq >> 3) : 7 - (sq >> 3);
                        if (!(passedMask[c][sq] & enP)) {
                            int bm = PASSED_MG[rr], be = PASSED_EG[rr];
                            int front = c == WHITE ? sq + 8 : sq - 8;
                            if (p.all & bit(front)) { bm /= 2; be /= 2; }
                            mg[c] += bm; eg[c] += be;
                            if (OPT_EVX) {
                                // king proximity to the stop square matters a lot in endgames
                                int w = max(0, 5 * rr - 13);
                                if (w) eg[c] += (cheb(ksq[them], front) * 19 / 4 - cheb(ksq[c], front) * 2) * w / 3;
                                // rule of the square: enemy has no pieces and the king cannot catch the pawn
                                if (!(p.bb[them][KNIGHT] | p.bb[them][BISHOP] | p.bb[them][ROOK] | p.bb[them][QUEEN]) &&
                                    !(frontFile[c][sq] & p.all)) {
                                    int promoSq = f + (c == WHITE ? 56 : 0);
                                    int dp = 7 - rr; if (rr == 1) dp--;
                                    int kd = cheb(ksq[them], promoSq) - (p.stm == them ? 1 : 0);
                                    if (dp < kd) eg[c] += 300 + 20 * rr;
                                }
                            }
                        }
                        if (frontFile[c][sq] & ownP) { mg[c] -= 10; eg[c] -= 20; }
                        if (!(adjFiles[f] & ownP)) { mg[c] -= 12; eg[c] -= 14; }
                        else if (pawnAtt[them][sq] & ownP) { mg[c] += 6; eg[c] += 4; }
                        break;
                    }
                    case KNIGHT: {
                        a = knightAtt[sq];
                        int mob = popcnt(a & ~own & ~pawnAttBB[them]);
                        mg[c] += 4 * (mob - 4); eg[c] += 4 * (mob - 4);
                        if (OPT_EVX) {
                            // outpost: advanced, pawn-supported and no enemy pawn can ever attack it
                            int rr = c == WHITE ? (sq >> 3) : 7 - (sq >> 3), f = sq & 7;
                            if (rr >= 3 && rr <= 5 && (pawnAtt[them][sq] & ownP) && !(passedMask[c][sq] & adjFiles[f] & enP)) {
                                mg[c] += 18; eg[c] += 10;
                            }
                        }
                        break;
                    }
                    case BISHOP: {
                        a = bishopAtt(sq, p.all);
                        int mob = popcnt(a & ~own & ~pawnAttBB[them]);
                        mg[c] += 3 * (mob - 6); eg[c] += 4 * (mob - 6);
                        break;
                    }
                    case ROOK: {
                        a = rookAtt(sq, p.all);
                        int mob = popcnt(a & ~own & ~pawnAttBB[them]);
                        mg[c] += 2 * (mob - 7); eg[c] += 4 * (mob - 7);
                        U64 fm = fileMask[sq & 7];
                        if (!(fm & ownP)) {
                            if (!(fm & enP)) { mg[c] += 22; eg[c] += 10; }
                            else { mg[c] += 10; eg[c] += 6; }
                        }
                        int rr = c == WHITE ? (sq >> 3) : 7 - (sq >> 3);
                        if (rr == 6) { mg[c] += 12; eg[c] += 18; }
                        break;
                    }
                    case QUEEN: {
                        a = bishopAtt(sq, p.all) | rookAtt(sq, p.all);
                        int mob = popcnt(a & ~own & ~pawnAttBB[them]);
                        mg[c] += (mob - 14); eg[c] += 2 * (mob - 14);
                        break;
                    }
                    default: break;
                }
                if (t >= KNIGHT && t <= QUEEN && (a & kzone[them])) {
                    atkCnt[c]++; atkUnits[c] += ATK_W[t] * popcnt(a & kzone[them]);
                }
            }
        }
        if (popcnt(p.bb[c][BISHOP]) >= 2) { mg[c] += 30; eg[c] += 50; }
    }
    // king safety
    for (int c = 0; c < 2; c++) {
        int them = c ^ 1, ks = ksq[c], kf = ks & 7, kr = ks >> 3;
        U64 ownP = p.bb[c][PAWN], enP = p.bb[them][PAWN];
        U64 files3 = fileMask[kf] | adjFiles[kf], r1, r2;
        if (c == WHITE) { r1 = kr < 7 ? (RANK_1 << (8 * (kr + 1))) : 0; r2 = kr < 6 ? (RANK_1 << (8 * (kr + 2))) : 0; }
        else { r1 = kr > 0 ? (RANK_1 << (8 * (kr - 1))) : 0; r2 = kr > 1 ? (RANK_1 << (8 * (kr - 2))) : 0; }
        mg[c] += 9 * popcnt(files3 & r1 & ownP) + 5 * popcnt(files3 & r2 & ownP);
        for (int ff = max(0, kf - 1); ff <= min(7, kf + 1); ff++) {
            if (!(fileMask[ff] & ownP)) { mg[c] -= 16; if (!(fileMask[ff] & enP)) mg[c] -= 10; }
        }
        if (atkCnt[them] >= 2) {
            int u = atkUnits[them];
            int pen = min(u * u / 8, 400);
            if (!p.bb[them][QUEEN]) pen /= 3;
            mg[c] -= pen;
        }
    }
    phase = min(phase, 24);
    int mgS = mg[0] - mg[1], egS = eg[0] - eg[1];
    if (OPT_EVX) {
        int npm[2], pcnt[2];
        for (int c = 0; c < 2; c++) {
            npm[c] = 3 * popcnt(p.bb[c][KNIGHT]) + 3 * popcnt(p.bb[c][BISHOP]) + 5 * popcnt(p.bb[c][ROOK]) + 9 * popcnt(p.bb[c][QUEEN]);
            pcnt[c] = popcnt(p.bb[c][PAWN]);
        }
        // mop-up: drive a pawnless defender to the edge and bring the king closer
        for (int c = 0; c < 2; c++) {
            int them = c ^ 1;
            if (!pcnt[them] && npm[c] - npm[them] >= 5 && npm[them] <= 3) {
                int k = ksq[them], f = k & 7, r = k >> 3;
                int cmd = max(3 - f, f - 4) + max(3 - r, r - 4);
                int bonus = 20 * cmd + 8 * (7 - cheb(ksq[0], ksq[1]));
                if (c == WHITE) egS += bonus; else egS -= bonus;
            }
        }
        // drawish endgames
        bool ocb = popcnt(p.bb[0][BISHOP]) == 1 && popcnt(p.bb[1][BISHOP]) == 1 &&
                   !(p.bb[0][KNIGHT] | p.bb[1][KNIGHT] | p.bb[0][ROOK] | p.bb[1][ROOK] | p.bb[0][QUEEN] | p.bb[1][QUEEN]) &&
                   ((p.bb[0][BISHOP] & LIGHT_SQ) != 0) != ((p.bb[1][BISHOP] & LIGHT_SQ) != 0);
        if (ocb) egS = egS * 5 / 8;
        else {
            int lead = egS > 0 ? 0 : 1;
            if (!pcnt[lead] && npm[lead] - npm[lead ^ 1] <= 3 && npm[lead ^ 1] > 0) egS /= 6;
        }
    }
    int score = (mgS * phase + egS * (24 - phase)) / 24;
    return (p.stm == WHITE ? score : -score) + 10;
}

static int nnueEval(const Pos& p) {
    const Acc& a = p.acc[p.ai];
    const int us = p.stm;
    const int b = min(NN_BUCKETS - 1, (popcnt(p.all) - 2) / 4);
    const int16_t* wu = NN.ow[b];
    const int16_t* wt = wu + NN_HL;
    const int16_t* au = a.v[us];
    const int16_t* at = a.v[us ^ 1];
    int32_t sum = 0;
    for (int j = 0; j < NN_HL; j++) { int v = min(max((int)au[j], 0), NN_QA); sum += v * v * wu[j]; }
    for (int j = 0; j < NN_HL; j++) { int v = min(max((int)at[j], 0), NN_QA); sum += v * v * wt[j]; }
    return (sum / NN_QA + NN.ob[b]) * NN_SCALE / (NN_QA * NN_QB);
}

// static evaluation from the side to move's point of view: NNUE when a net is active (p.track), else HCE
static int evaluate(const Pos& p) {
    int s;
    if (p.track) {
        if (!p.bb[0][PAWN] && !p.bb[1][PAWN] && !(p.bb[0][ROOK] | p.bb[1][ROOK] | p.bb[0][QUEEN] | p.bb[1][QUEEN]) &&
            popcnt(p.bb[0][KNIGHT] | p.bb[0][BISHOP] | p.bb[1][KNIGHT] | p.bb[1][BISHOP]) <= 1) return 0;
        s = nnueEval(p);
    } else s = evaluateHCE(p);
    s = s * (200 - min(p.half, 100)) / 200;   // fade towards a draw as the 50-move counter grows
    return max(-MATE_BOUND + 1, min(MATE_BOUND - 1, s));
}

// ---------------------------------------------------------------- transposition table
// 4-way bucketed (64-byte aligned), depth/age based replacement.
enum { TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3 };
// decoded view of a table slot
struct TTEntry { int32_t move; int16_t score; int8_t depth; uint8_t fa; };
// A slot is two 64-bit words: data and (key ^ data). Torn reads from other threads fail the check and are
// treated as a miss, so the table can be shared lock-free between search threads (Lazy SMP).
struct TTSlot { atomic<U64> keyx, data; };
static inline U64 ttPack(const TTEntry& e) {
    return (U64)(uint32_t)e.move | ((U64)(uint16_t)e.score << 32) | ((U64)(uint8_t)e.depth << 48) | ((U64)e.fa << 56);
}
static inline TTEntry ttUnpack(U64 d) {
    TTEntry e; e.move = (int32_t)(uint32_t)d; e.score = (int16_t)(uint16_t)(d >> 32);
    e.depth = (int8_t)(uint8_t)(d >> 48); e.fa = (uint8_t)(d >> 56); return e;
}
struct TT {
    unsigned char* raw = nullptr; TTSlot* t = nullptr; size_t n = 0; U64 mask = 0; uint8_t age = 0;
    ~TT() { free(raw); }
    void resize(size_t mb) {
        free(raw); raw = nullptr;
        size_t cnt = 4;
        while (cnt * 2 * sizeof(TTSlot) <= mb * 1024 * 1024) cnt *= 2;
        n = cnt;
        raw = (unsigned char*)malloc(n * sizeof(TTSlot) + 64);
        t = (TTSlot*)(((uintptr_t)raw + 63) & ~(uintptr_t)63);
        memset((void*)t, 0, n * sizeof(TTSlot));
        mask = n / 4 - 1;
    }
    void clear() { memset((void*)t, 0, n * sizeof(TTSlot)); }
    bool probe(U64 k, TTEntry& out) const {
        const TTSlot* b = t + (k & mask) * 4;
        for (int i = 0; i < 4; i++) {
            U64 d = b[i].data.load(memory_order_relaxed), kx = b[i].keyx.load(memory_order_relaxed);
            if ((kx ^ d) == k && (d >> 56) != 0) { out = ttUnpack(d); return true; }
        }
        return false;
    }
    void store(U64 k, Move m, int score, int depth, int flag, int ply) {
        TTSlot* b = t + (k & mask) * 4;
        TTSlot* rep = nullptr; int repVal = 1 << 30; bool same = false; TTEntry old{};
        for (int i = 0; i < 4; i++) {
            U64 d = b[i].data.load(memory_order_relaxed), kx = b[i].keyx.load(memory_order_relaxed);
            TTEntry c = ttUnpack(d);
            if (c.fa == 0) { rep = &b[i]; same = false; break; }
            if ((kx ^ d) == k) { rep = &b[i]; same = true; old = c; break; }
            int val = c.depth - 6 * ((age - (c.fa >> 2)) & 63);
            if (val < repVal) { repVal = val; rep = &b[i]; }
        }
        if (score >= MATE_BOUND) score += ply; else if (score <= -MATE_BOUND) score -= ply;
        if (same) {
            if (m == NOMOVE) m = old.move;
            if (flag != TT_EXACT && depth + 4 <= old.depth && (old.fa >> 2) == age) return;
        }
        TTEntry e; e.move = m; e.score = (int16_t)score; e.depth = (int8_t)depth; e.fa = (uint8_t)(flag | (age << 2));
        U64 d = ttPack(e);
        rep->data.store(d, memory_order_relaxed); rep->keyx.store(d ^ k, memory_order_relaxed);
    }
};
static TT tt;
static int fromTT(int s, int ply) { if (s >= MATE_BOUND) return s - ply; if (s <= -MATE_BOUND) return s + ply; return s; }

// ---------------------------------------------------------------- search
static atomic<bool> g_stop(false), g_helperStop(false);
static int lmrTable[64][64];

static inline void updC(int16_t& h, int bonus) {
    int v = h; v += bonus - v * abs(bonus) / 16384; h = (int16_t)v;
}

struct Searcher;
static vector<Searcher*> g_helperPtrs;
static int g_threads = 1;

struct Searcher {
    Pos pos;
    U64 nodes = 0, nodeLimit = 0;
    bool stopped = false, timed = false;
    int64_t softMs = 0, hardMs = 0;
    int maxDepth = 99, completed = 0;
    bool quiet = false;
    Move bestOut = NOMOVE; int scoreOut = 0;
    chrono::steady_clock::time_point t0;
    int staticEval[MAXPLY + 4];
    Move killers[MAXPLY + 4][2];
    Move mvStack[MAXPLY + 4], excl[MAXPLY + 4];
    int pcStack[MAXPLY + 4];
    Move rootMove = NOMOVE;
    int history[2][64][64];
    int corrHist[2][16384];
    int threadId = 0;
    int16_t cont1[6][64][6][64], cont2[6][64][6][64];
    Move counterMove[64][64];
    Move pv[MAXPLY + 4][MAXPLY + 4];
    int pvLen[MAXPLY + 4];

    U64 totalNodes() const { U64 t = nodes; if (threadId == 0) for (auto* h : g_helperPtrs) t += h->nodes; return t; }
    int64_t elapsed() const { return chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t0).count(); }
    void checkTime() {
        if (completed < 1) return;
        if (g_stop.load() || (threadId > 0 && g_helperStop.load())) { stopped = true; return; }
        if (nodeLimit && nodes >= nodeLimit) { stopped = true; return; }
        if (timed && elapsed() >= hardMs) stopped = true;
    }
    void clearHistory() {
        memset(history, 0, sizeof(history)); memset(corrHist, 0, sizeof(corrHist)); memset(cont1, 0, sizeof(cont1));
        memset(cont2, 0, sizeof(cont2)); memset(counterMove, 0, sizeof(counterMove));
    }

    void updateHist(int us, Move m, int bonus) {
        int& h = history[us][mFrom(m)][mTo(m)];
        h += bonus - h * abs(bonus) / 16384;
    }
    // history score of a quiet move: main history + 1-ply and 2-ply continuation history
    int histScore(Move m, int ply) const {
        int from = mFrom(m), to = mTo(m), pc = pos.pt[from];
        int s = history[pos.stm][from][to];
        if (OPT_CH) {
            if (ply >= 1 && mvStack[ply - 1]) s += cont1[pcStack[ply - 1]][mTo(mvStack[ply - 1])][pc][to];
            if (ply >= 2 && mvStack[ply - 2]) s += cont2[pcStack[ply - 2]][mTo(mvStack[ply - 2])][pc][to];
        }
        return s;
    }
    void updQuiet(int us, Move m, int ply, int bonus) {
        updateHist(us, m, bonus);
        if (OPT_CH) {
            int from = mFrom(m), to = mTo(m), pc = pos.pt[from];
            if (ply >= 1 && mvStack[ply - 1]) updC(cont1[pcStack[ply - 1]][mTo(mvStack[ply - 1])][pc][to], bonus);
            if (ply >= 2 && mvStack[ply - 2]) updC(cont2[pcStack[ply - 2]][mTo(mvStack[ply - 2])][pc][to], bonus);
        }
    }

    int scoreMove(Move m, Move ttMove, int ply) {
        if (m == ttMove) return 1 << 30;
        int from = mFrom(m), to = mTo(m), promo = mPromo(m);
        if (m & (F_CAP | F_EP)) {
            int victim = (m & F_EP) ? PAWN : pos.pt[to], attacker = pos.pt[from];
            int s = victim * 8 - attacker;
            if (promo == QUEEN) s += 50;
            if (attacker > victim && !promo && pos.see(m) < 0) return -1000000 + s;
            return 2000000 + s;
        }
        if (promo) return promo == QUEEN ? 1900000 : -500000;
        if (m == killers[ply][0]) return 1500000;
        if (m == killers[ply][1]) return 1400000;
        if (OPT_CH && ply >= 1 && mvStack[ply - 1] && m == counterMove[mFrom(mvStack[ply - 1])][mTo(mvStack[ply - 1])])
            return 1300000;
        return histScore(m, ply);
    }

    int qsearch(int alpha, int beta, int ply) {
        pvLen[ply] = ply;
        if ((nodes & 2047) == 0) checkTime();
        if (stopped) return 0;
        nodes++;
        if (ply >= MAXPLY - 1) return evaluate(pos);
        bool pvNode = beta - alpha > 1;
        bool inCheck = OPT_TTQ ? pos.inCheck() : false;
        int origAlpha = alpha;
        Move ttm = NOMOVE;
        if (OPT_TTQ) {
            TTEntry te;
            if (tt.probe(pos.key, te)) {
                ttm = te.move;
                if (!pvNode) {
                    int s = fromTT(te.score, ply), fl = te.fa & 3;
                    if (fl == TT_EXACT || (fl == TT_LOWER && s >= beta) || (fl == TT_UPPER && s <= alpha)) return s;
                }
            }
        }
        int stand = -INF;
        if (!inCheck) {
            stand = evaluate(pos);
            if (stand >= beta) return stand;
            if (stand > alpha) alpha = stand;
        }
        Move mv[256]; int sc[256];
        int n = pos.gen(mv, !inCheck);
        for (int i = 0; i < n; i++) {
            Move m = mv[i];
            int victim = (m & F_EP) ? PAWN : (pos.pt[mTo(m)] == NO_PIECE ? PAWN : pos.pt[mTo(m)]);
            sc[i] = victim * 8 - pos.pt[mFrom(m)] + (mPromo(m) ? 50 : 0);
            if (m == ttm) sc[i] += 10000;
        }
        int best = inCheck ? -INF : stand, legal = 0;
        Move bm = NOMOVE;
        static const int V[6] = {100, 320, 330, 500, 900, 0};
        for (int i = 0; i < n; i++) {
            int bi = i;
            for (int j = i + 1; j < n; j++) if (sc[j] > sc[bi]) bi = j;
            swap(mv[i], mv[bi]); swap(sc[i], sc[bi]);
            Move m = mv[i];
            if (!inCheck && !mPromo(m)) {
                int victim = (m & F_EP) ? PAWN : pos.pt[mTo(m)];
                if (stand + 200 + V[victim] < alpha) continue;
                if (pos.see(m) < 0) continue;
            }
            pos.make(m);
            if (pos.illegalAfterMake()) { pos.unmake(); continue; }
            legal++;
            int score = -qsearch(-beta, -alpha, ply + 1);
            pos.unmake();
            if (stopped) return 0;
            if (score > best) {
                best = score; bm = m;
                if (score > alpha) { alpha = score; if (score >= beta) break; }
            }
        }
        if (inCheck && legal == 0) return -MATE + ply;
        if (OPT_TTQ) {
            int flag = best >= beta ? TT_LOWER : (best > origAlpha ? TT_EXACT : TT_UPPER);
            tt.store(pos.key, bm, best, 0, flag, ply);
        }
        return best;
    }

    int negamax(int depth, int alpha, int beta, int ply, bool doNull, bool cutNode = false) {
        pvLen[ply] = ply;
        if ((nodes & 2047) == 0) checkTime();
        if (stopped) return 0;
        bool pvNode = beta - alpha > 1;
        if (ply > 0) {
            if (pos.half >= 100 || pos.repetition()) return 0;
            alpha = max(alpha, -MATE + ply); beta = min(beta, MATE - ply - 1);
            if (alpha >= beta) return alpha;
        }
        if (ply >= MAXPLY - 1) return evaluate(pos);
        bool inCheck = pos.inCheck();
        if (inCheck) depth++;
        if (depth <= 0) return qsearch(alpha, beta, ply);
        nodes++;

        int us = pos.stm;
        Move excluded = excl[ply];
        TTEntry te;
        bool ttHit = tt.probe(pos.key, te);
        Move ttMove = NOMOVE; int ttScore = 0, ttDepth = -1, ttFlag = 0;
        if (ttHit) {
            ttMove = te.move; ttScore = fromTT(te.score, ply); ttDepth = te.depth; ttFlag = te.fa & 3;
            if (!pvNode && ply > 0 && !excluded && ttDepth >= depth) {
                if (ttFlag == TT_EXACT) return ttScore;
                if (ttFlag == TT_LOWER && ttScore >= beta) return ttScore;
                if (ttFlag == TT_UPPER && ttScore <= alpha) return ttScore;
            }
        }
        int eval = VALUE_NONE;
        int corrIdx = 0;
        if (!inCheck) {
            eval = evaluate(pos);
            // pawn-structure keyed correction of the static eval (learns systematic eval error)
            corrIdx = (int)(((pos.bb[0][PAWN] * 0x9E3779B97F4A7C15ULL) ^ (pos.bb[1][PAWN] * 0xC2B2AE3D27D4EB4FULL)) >> 50);
            if (OPT_CORR) eval += corrHist[us][corrIdx] / 256;
            eval = max(-MATE_BOUND + 1, min(MATE_BOUND - 1, eval));
        }
        staticEval[ply] = eval;
        bool improving = !inCheck && ply >= 2 && staticEval[ply - 2] != VALUE_NONE && eval > staticEval[ply - 2];
        int pEval = eval;   // eval refined by the transposition table bound
        if (!inCheck && ttHit && abs(ttScore) < MATE_BOUND) {
            if (ttFlag == TT_EXACT || (ttFlag == TT_LOWER && ttScore > eval) || (ttFlag == TT_UPPER && ttScore < eval))
                pEval = ttScore;
        }

        if (!pvNode && !inCheck) {
            if (depth <= 6 && pEval - 85 * depth >= beta && abs(beta) < MATE_BOUND) return pEval;
            if (depth <= 2 && pEval + 250 * depth <= alpha) {
                int q = qsearch(alpha, beta, ply);
                if (q <= alpha) return q;
            }
            if (doNull && !excluded && depth >= 3 && pEval >= beta &&
                (pos.bb[us][KNIGHT] | pos.bb[us][BISHOP] | pos.bb[us][ROOK] | pos.bb[us][QUEEN])) {
                int R = 3 + depth / 4 + min((pEval - beta) / 200, 3);
                mvStack[ply] = NOMOVE; pcStack[ply] = 0;
                pos.makeNull();
                int s = -negamax(depth - 1 - R, -beta, -beta + 1, ply + 1, false, !cutNode);
                pos.unmakeNull();
                if (stopped) return 0;
                if (s >= beta) return s >= MATE_BOUND ? beta : s;
            }
        }
        if (depth >= 4 && ttMove == NOMOVE) depth--;
        if (OPT_CUT && cutNode && depth >= 8 && ttMove == NOMOVE) depth--;

        // ProbCut: a good capture that still beats beta by a margin at reduced depth very likely fails high
        if (OPT_PC && !pvNode && !inCheck && !excluded && depth >= 5 && abs(beta) < MATE_BOUND) {
            int pcBeta = beta + 190;
            if (!(ttHit && ttDepth >= depth - 3 && ttScore < pcBeta)) {
                Move cm[256]; int cs[256];
                int cn = pos.gen(cm, true);
                for (int i = 0; i < cn; i++) {
                    int victim = (cm[i] & F_EP) ? PAWN : (pos.pt[mTo(cm[i])] == NO_PIECE ? PAWN : pos.pt[mTo(cm[i])]);
                    cs[i] = victim * 8 - pos.pt[mFrom(cm[i])] + (cm[i] == ttMove ? 1000 : 0);
                }
                for (int i = 0; i < cn; i++) {
                    int bi = i;
                    for (int j = i + 1; j < cn; j++) if (cs[j] > cs[bi]) bi = j;
                    swap(cm[i], cm[bi]); swap(cs[i], cs[bi]);
                    Move m = cm[i];
                    if (pos.see(m) < pcBeta - eval) continue;
                    mvStack[ply] = m; pcStack[ply] = pos.pt[mFrom(m)];
                    pos.make(m);
                    if (pos.illegalAfterMake()) { pos.unmake(); continue; }
                    int sc2 = -qsearch(-pcBeta, -pcBeta + 1, ply + 1);
                    if (sc2 >= pcBeta) sc2 = -negamax(depth - 4, -pcBeta, -pcBeta + 1, ply + 1, true, !cutNode);
                    pos.unmake();
                    if (stopped) return 0;
                    if (sc2 >= pcBeta) {
                        tt.store(pos.key, m, sc2, depth - 3, TT_LOWER, ply);
                        return sc2 - (pcBeta - beta);
                    }
                }
            }
        }

        Move mv[256]; int sc[256];
        int n = pos.gen(mv, false);
        for (int i = 0; i < n; i++) sc[i] = scoreMove(mv[i], ttMove, ply);

        int bestScore = -INF, origAlpha = alpha, legal = 0, quietCount = 0;
        Move bestMove = NOMOVE, quiets[64];
        bool ttCap = ttMove != NOMOVE && (ttMove & (F_CAP | F_EP));
        for (int i = 0; i < n; i++) {
            int bi = i;
            for (int j = i + 1; j < n; j++) if (sc[j] > sc[bi]) bi = j;
            swap(mv[i], mv[bi]); swap(sc[i], sc[bi]);
            Move m = mv[i];
            if (m == excluded) continue;
            bool isCap = (m & (F_CAP | F_EP)) != 0, isPromo = mPromo(m) != 0, quiet = !isCap && !isPromo;
            int hs = quiet ? histScore(m, ply) : 0;
            if (!pvNode && !inCheck && legal > 0 && bestScore > -MATE_BOUND) {
                if (isCap && !isPromo && depth <= 6 && sc[i] < 2000000 && pos.see(m) < -90 * depth) continue;
                if (OPT_PR && quiet && depth <= 6 && pos.see(m) < -50 * depth) continue;
                if (OPT_PR && quiet && depth <= 4 && hs < -2500 * depth) continue;
            }
            // singular extension: is the TT move much better than all alternatives?
            int ext = 0;
            if (OPT_SE && m == ttMove && !excluded && ply > 0 && depth >= 8 && ttDepth >= depth - 3 &&
                (ttFlag == TT_LOWER || ttFlag == TT_EXACT) && abs(ttScore) < MATE_BOUND) {
                int sBeta = ttScore - 2 * depth;
                excl[ply] = m;
                int s = negamax((depth - 1) / 2, sBeta - 1, sBeta, ply, false);
                excl[ply] = NOMOVE;
                pvLen[ply] = ply;
                if (stopped) return 0;
                if (s < sBeta) ext = 1;
                else if (sBeta >= beta) return sBeta;
                else if (ttScore >= beta) ext = -1;
            }
            mvStack[ply] = m; pcStack[ply] = pos.pt[mFrom(m)];
            pos.make(m);
            if (pos.illegalAfterMake()) { pos.unmake(); continue; }
            legal++;
            bool givesCheck = pos.inCheck();
            if (quiet && !pvNode && !inCheck && !givesCheck && bestScore > -MATE_BOUND) {
                if (depth <= 6 && pEval + 90 + 80 * depth <= alpha) { pos.unmake(); continue; }
                int lmpLimit = improving ? 4 + depth * depth : (4 + depth * depth) / 2 + 1;
                if (depth <= 4 && quietCount >= lmpLimit) { pos.unmake(); continue; }
            }
            int newDepth = depth - 1 + ext, score;
            if (legal == 1) score = -negamax(newDepth, -beta, -alpha, ply + 1, true, false);
            else {
                int r = 0;
                if (depth >= 3 && quiet && legal >= (pvNode ? 4 : 3)) {
                    r = lmrTable[min(depth, 63)][min(legal, 63)];
                    if (!pvNode) r++;
                    if (!improving) r++;
                    if (OPT_CUT && cutNode) r++;
                    if (givesCheck) r--;
                    if (m == killers[ply][0] || m == killers[ply][1]) r--;
                    if (ttCap) r++;
                    r -= max(-2, min(2, hs / 5000));
                    r = max(0, min(r, newDepth - 1));
                }
                score = -negamax(newDepth - r, -alpha - 1, -alpha, ply + 1, true, true);
                if (score > alpha && r > 0) score = -negamax(newDepth, -alpha - 1, -alpha, ply + 1, true, !cutNode);
                if (score > alpha && score < beta) score = -negamax(newDepth, -beta, -alpha, ply + 1, true, false);
            }
            pos.unmake();
            if (stopped) return 0;
            if (score > bestScore) {
                bestScore = score; bestMove = m;
                if (score > alpha) {
                    alpha = score;
                    if (ply == 0) rootMove = m;
                    pv[ply][ply] = m;
                    for (int j = ply + 1; j < pvLen[ply + 1]; j++) pv[ply][j] = pv[ply + 1][j];
                    pvLen[ply] = max(pvLen[ply + 1], ply + 1);
                    if (score >= beta) {
                        if (quiet) {
                            if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                            int bonus = min(depth * depth, 1600);
                            updQuiet(us, m, ply, bonus);
                            for (int j = 0; j < quietCount; j++) updQuiet(us, quiets[j], ply, -bonus);
                            if (ply >= 1 && mvStack[ply - 1]) counterMove[mFrom(mvStack[ply - 1])][mTo(mvStack[ply - 1])] = m;
                        }
                        break;
                    }
                }
            }
            if (quiet && quietCount < 64) quiets[quietCount++] = m;
        }
        if (legal == 0) return excluded ? alpha : (inCheck ? -MATE + ply : 0);
        if (!excluded) {
            int flag = bestScore >= beta ? TT_LOWER : (bestScore > origAlpha ? TT_EXACT : TT_UPPER);
            tt.store(pos.key, bestMove, bestScore, depth, flag, ply);
            if (OPT_CORR && !inCheck && !(bestMove && (bestMove & (F_CAP | F_EP))) && abs(bestScore) < MATE_BOUND &&
                !(flag == TT_LOWER && bestScore <= eval) && !(flag == TT_UPPER && bestScore >= eval)) {
                int diff = max(-200, min(200, bestScore - eval)), w = min(depth + 1, 16);
                int& c = corrHist[us][corrIdx];
                c += ((diff * 256 - c) * w) / 128;
            }
        }
        return bestScore;
    }

    void think() {
        nodes = 0; stopped = false; completed = 0; t0 = chrono::steady_clock::now();
        memset(killers, 0, sizeof(killers)); memset(mvStack, 0, sizeof(mvStack));
        memset(excl, 0, sizeof(excl)); memset(pcStack, 0, sizeof(pcStack));
        for (int i = 0; i < MAXPLY + 4; i++) { pvLen[i] = i; staticEval[i] = VALUE_NONE; }
        if (threadId == 0) tt.age = (tt.age + 1) & 63;
        pos.track = false; pos.ai = 0;
        if (NN.loaded && g_useNNUE) { pos.track = true; pos.refreshAcc(); }
        bestOut = NOMOVE; scoreOut = 0;
        Move legal[256];
        int n = legalMoves(pos, legal);
        if (n == 0) { if (!quiet) { printf("bestmove 0000\n"); fflush(stdout); } return; }
        Move best = legal[0], lastBest = NOMOVE;
        int prevScore = 0, stable = 0;
        static const int skipSize[20] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
        static const int skipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7};
        for (int depth = 1; depth <= maxDepth; depth++) {
            if (threadId > 0 && depth > 1) {
                int k = (threadId - 1) % 20;
                if (((depth + skipPhase[k]) / skipSize[k]) % 2) continue;
            }
            int delta = 25, alpha = -INF, beta = INF, score;
            if (depth >= 5) { alpha = max(prevScore - delta, -INF); beta = min(prevScore + delta, INF); }
            rootMove = NOMOVE;
            for (;;) {
                score = negamax(depth, alpha, beta, 0, false);
                if (stopped) break;
                if (score <= alpha) { beta = (alpha + beta) / 2; alpha = max(score - delta, -INF); }
                else if (score >= beta) { beta = min(score + delta, INF); }
                else break;
                delta += delta / 2;
            }
            if (stopped) {
                // keep the best move of the unfinished iteration if one was fully searched
                if (rootMove != NOMOVE) best = rootMove;
                break;
            }
            int oldScore = prevScore;
            completed = depth; prevScore = score;
            if (pvLen[0] > 0) best = pv[0][0];
            if (best == lastBest) stable++; else { stable = 0; lastBest = best; }
            int64_t ms = elapsed();
            if (!quiet) {
            printf("info depth %d score ", depth);
            if (abs(score) >= MATE_BOUND) printf("mate %d", (score > 0 ? 1 : -1) * ((MATE - abs(score) + 1) / 2));
            else printf("cp %d", score);
            printf(" nodes %llu nps %llu time %lld pv", (unsigned long long)totalNodes(),
                   (unsigned long long)(totalNodes() * 1000 / max<int64_t>(1, ms)), (long long)ms);
            for (int i = 0; i < pvLen[0]; i++) printf(" %s", moveStr(pv[0][i]).c_str());
            printf("\n"); fflush(stdout);
            }
            if (n == 1 && depth >= 4) break;
            if (timed) {
                double f = stable >= 6 ? 0.45 : (stable >= 3 ? 0.55 : 0.65);
                if (depth >= 6 && oldScore - score >= 20) f = 1.0;   // score dropped: think longer
                if (ms > softMs * f) break;
            }
            if (abs(score) >= MATE_BOUND && depth >= 2 && (MATE - abs(score)) < depth) break;
        }
        bestOut = best; scoreOut = prevScore;
        if (!quiet) { printf("bestmove %s\n", moveStr(best).c_str()); fflush(stdout); }
    }
};
static Searcher S;

static void initSearch() {
    for (int d = 0; d < 64; d++) for (int m = 0; m < 64; m++)
        lmrTable[d][m] = (d >= 1 && m >= 1) ? (int)(0.75 + log((double)d) * log((double)m) / 2.25) : 0;
}


// ---------------------------------------------------------------- self-play data generation (for NNUE training)
// Writes lines "fen;score_cp_white_pov;result_white_pov" (result 1 / 0.5 / 0). Quiet positions only.
static int gameState(Pos& p, int& nLegal) {   // 0 ongoing, 1 mate (stm lost), 2 draw
    Move mv[256]; nLegal = legalMoves(p, mv);
    if (nLegal == 0) return p.inCheck() ? 1 : 2;
    if (!p.bb[0][PAWN] && !p.bb[1][PAWN] && !(p.bb[0][ROOK] | p.bb[1][ROOK] | p.bb[0][QUEEN] | p.bb[1][QUEEN]) && popcnt(p.all) <= 3) return 2;
    if (p.half >= 100) return 2;
    int reps = 0;
    for (int i = 2; i <= min(p.half, p.hp); i += 2) if (p.hist[p.hp - i].key == p.key) reps++;
    if (reps >= 2) return 2;
    return 0;
}

static void datagen(int games, U64 nodes, int maxDepth, const string& outFile, U64 seed, int randPlies) {
    RNG rng(seed * 0x9E3779B97F4A7C15ULL + 12345);
    for (int i = 0; i < 8; i++) rng.next();
    FILE* out = fopen(outFile.c_str(), "a");
    if (!out) { printf("cannot open %s\n", outFile.c_str()); fflush(stdout); return; }
    static Pos g;
    S.quiet = true;
    long long positions = 0; int done = 0;
    auto t0 = chrono::steady_clock::now();
    for (int gi = 0; done < games; gi++) {
        g.setFen(START_FEN);
        bool ok = true;
        for (int k = 0; k < randPlies + (int)(rng.next() & 1); k++) {
            Move mv[256]; int n = legalMoves(g, mv);
            if (!n) { ok = false; break; }
            g.make(mv[rng.next() % n]);
        }
        int nl; if (!ok || gameState(g, nl)) continue;
        // reject lopsided openings
        S.pos = g; S.maxDepth = 8; S.nodeLimit = 0; S.timed = false; S.think();
        if (abs(S.scoreOut) > 300) continue;
        tt.clear(); S.clearHistory();
        vector<string> lines; vector<int> sc; vector<int> stmv;
        double result = 0.5; int winStreak = 0, drawStreak = 0, plies = 0;
        for (;;) {
            int n; int st = gameState(g, n);
            if (st == 1) { result = g.stm == WHITE ? 0.0 : 1.0; break; }
            if (st == 2) { result = 0.5; break; }
            S.pos = g; S.maxDepth = maxDepth; S.nodeLimit = nodes; S.timed = false;
            S.think();
            Move best = S.bestOut; int score = S.scoreOut;   // side to move's view
            if (best == NOMOVE) { result = 0.5; break; }
            int wscore = g.stm == WHITE ? score : -score;
            bool noisy = (best & (F_CAP | F_EP)) || mPromo(best);
            if (!g.inCheck() && !noisy && abs(score) < 2500) {
                lines.push_back(g.fen()); sc.push_back(wscore); stmv.push_back(g.stm);
            }
            if (abs(score) >= 1000) { winStreak++; if (winStreak >= 5) { result = wscore > 0 ? 1.0 : 0.0; break; } } else winStreak = 0;
            if (abs(score) <= 8 && plies > 100) { drawStreak++; if (drawStreak >= 12) { result = 0.5; break; } } else drawStreak = 0;
            g.make(best); plies++;
            if (plies > 400) { result = 0.5; break; }
        }
        for (size_t i = 0; i < lines.size(); i++) fprintf(out, "%s;%d;%.1f\n", lines[i].c_str(), sc[i], result);
        positions += (long long)lines.size(); done++;
        if (done % 10 == 0) {
            fflush(out);
            double secs = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
            printf("datagen games %d positions %lld (%.0f pos/s)\n", done, positions, positions / max(secs, 0.001)); fflush(stdout);
        }
    }
    fclose(out);
    S.quiet = false;
    printf("datagen finished: %d games, %lld positions\n", done, positions); fflush(stdout);
}

static const char* BENCH_FENS[] = {
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "2rq1rk1/pp1bppbp/3p1np1/8/3NP3/2N1BP2/PPPQ2PP/2KR1B1R w - - 0 12",
    "8/8/4k3/8/2p5/2K5/8/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP2BPPP/R1BQK2R w KQ - 0 7",
};


// incremental NNUE accumulator must always equal a from-scratch refresh
static bool nnueSelfTest() {
    bool tmpNet = false;
    if (!NN.loaded) {   // random net just for the consistency check
        RNG r(99);
        for (auto& w : NN.fw) w = (int16_t)((int)(r.next() % 201) - 100);
        for (auto& w : NN.fb) w = (int16_t)((int)(r.next() % 201) - 100);
        for (auto& w : NN.ow[0]) w = 0; NN.loaded = true; tmpNet = true;
    }
    RNG rng(2024); long checks = 0; bool ok = true;
    for (int game = 0; game < 300 && ok; game++) {
        static Pos p; p.setFen(START_FEN); p.track = true; p.refreshAcc();
        vector<Acc> saved;
        for (int ply = 0; ply < 120 && ok; ply++) {
            Move mv[256]; int n = legalMoves(p, mv);
            if (!n) break;
            saved.push_back(p.acc[p.ai]);
            Move m = mv[rng.next() % n];
            p.make(m);
            Pos q; q.setFen(p.fen()); q.track = true; q.refreshAcc();   // fen round trip as well
            if (memcmp(&q.acc[0], &p.acc[p.ai], sizeof(Acc)) != 0) { printf("ACC MISMATCH game %d ply %d move %s\n", game, ply, moveStr(m).c_str()); ok = false; }
            if (q.key != p.key) { printf("KEY MISMATCH after fen round trip\n"); ok = false; }
            checks++;
            if (rng.next() % 5 == 0) { p.unmake(); if (memcmp(&saved.back(), &p.acc[p.ai], sizeof(Acc)) != 0) { printf("UNMAKE MISMATCH\n"); ok = false; } saved.pop_back(); }
        }
    }
    if (tmpNet) NN.loaded = false;
    printf("selftest %s (%ld positions checked)\n", ok ? "PASSED" : "FAILED", checks); fflush(stdout);
    return ok;
}

// ---------------------------------------------------------------- UCI
int main() {
    initTables(); initEval(); initSearch();
#ifdef CF_HAVE_EMBEDDED
    nnLoadBuf(CF_NET, sizeof(CF_NET), "embedded");
#endif
    tt.resize(64);
    S.clearHistory();
    int moveOverhead = 30;
    static Pos root;
    root.setFen(START_FEN);
    thread worker;
    auto joinWorker = [&]() { if (worker.joinable()) worker.join(); };
    string line;
    while (getline(cin, line)) {
        istringstream is(line);
        string cmd;
        is >> cmd;
        if (cmd == "uci") {
            printf("id name ClaudeFish 3.0\nid author Claude (Anthropic)\n");
            printf("option name Threads type spin default 1 min 1 max 64\n");
            printf("option name UseNNUE type check default false\n");
            printf("option name EvalFile type string default <embedded>\n");
            printf("option name Hash type spin default 64 min 1 max 4096\n");
            printf("option name Move Overhead type spin default 30 min 0 max 5000\n");
            printf("option name SE type spin default 1 min 0 max 1\n");
            printf("option name PC type spin default 0 min 0 max 1\n");
            printf("option name CORR type spin default 0 min 0 max 1\n");
            printf("option name CUT type spin default 0 min 0 max 1\n");
            printf("option name CH type spin default 1 min 0 max 1\n");
            printf("option name TTQ type spin default 1 min 0 max 1\n");
            printf("option name PR type spin default 1 min 0 max 1\n");
            printf("option name EVX type spin default 1 min 0 max 1\n");
            printf("uciok\n"); fflush(stdout);
        } else if (cmd == "isready") {
            printf("readyok\n"); fflush(stdout);
        } else if (cmd == "ucinewgame") {
            g_stop = true; joinWorker(); g_stop = false;
            tt.clear(); S.clearHistory();
            for (auto* h : g_helperPtrs) h->clearHistory();
        } else if (cmd == "setoption") {
            string tok, name, value;
            is >> tok;  // name
            while (is >> tok && tok != "value") name += (name.empty() ? "" : " ") + tok;
            is >> value;
            if (name == "Hash") { g_stop = true; joinWorker(); g_stop = false; tt.resize(max(1, atoi(value.c_str()))); }
            else if (name == "Move Overhead") moveOverhead = max(0, atoi(value.c_str()));
            else if (name == "Threads") { g_threads = max(1, min(64, atoi(value.c_str()))); }
            else if (name == "UseNNUE") g_useNNUE = (value == "true" || value == "1");
            else if (name == "EvalFile") {
                g_stop = true; joinWorker(); g_stop = false;
                if (!nnLoadFile(value)) { printf("info string failed to load net %s\n", value.c_str()); fflush(stdout); }
                else { printf("info string loaded net %s\n", g_netName.c_str()); fflush(stdout); }
            }
            else if (name == "PC") OPT_PC = atoi(value.c_str()) != 0;
            else if (name == "CORR") OPT_CORR = atoi(value.c_str()) != 0;
            else if (name == "CUT") OPT_CUT = atoi(value.c_str()) != 0;
            else if (name == "SE") OPT_SE = atoi(value.c_str()) != 0;
            else if (name == "CH") OPT_CH = atoi(value.c_str()) != 0;
            else if (name == "TTQ") OPT_TTQ = atoi(value.c_str()) != 0;
            else if (name == "PR") OPT_PR = atoi(value.c_str()) != 0;
            else if (name == "EVX") OPT_EVX = atoi(value.c_str()) != 0;
        } else if (cmd == "position") {
            g_stop = true; joinWorker(); g_stop = false;
            string tok;
            is >> tok;
            if (tok == "startpos") { root.setFen(START_FEN); if (!(is >> tok)) tok = ""; }
            else if (tok == "fen") {
                string fen; tok = "";
                while (is >> tok && tok != "moves") fen += tok + " ";
                root.setFen(fen);
            }
            if (tok == "moves") {
                string ms;
                while (is >> ms) { Move m = parseMove(root, ms); if (!m) break; root.make(m); }
            }
        } else if (cmd == "go") {
            g_stop = true; joinWorker(); g_stop = false;
            int wtime = -1, btime = -1, winc = 0, binc = 0, mtg = 0, movetime = -1, depth = 99;
            U64 nodesLim = 0; bool infinite = false;
            string tok;
            while (is >> tok) {
                if (tok == "wtime") is >> wtime; else if (tok == "btime") is >> btime;
                else if (tok == "winc") is >> winc; else if (tok == "binc") is >> binc;
                else if (tok == "movestogo") is >> mtg; else if (tok == "movetime") is >> movetime;
                else if (tok == "depth") is >> depth; else if (tok == "nodes") is >> nodesLim;
                else if (tok == "infinite") infinite = true;
            }
            S.pos = root; S.maxDepth = min(depth, 99); S.nodeLimit = nodesLim; S.timed = false;
            if (movetime > 0) {
                S.timed = true; S.hardMs = max(1, movetime - moveOverhead); S.softMs = S.hardMs;
            } else if (!infinite && depth == 99 && !nodesLim && (wtime >= 0 || btime >= 0)) {
                int tl = root.stm == WHITE ? wtime : btime, inc = root.stm == WHITE ? winc : binc;
                if (tl < 0) tl = 1000;
                int div = mtg > 0 ? min(mtg, 40) : 30;
                int64_t soft = tl / div + inc * 3 / 4;
                soft = min<int64_t>(soft, tl / 2);
                int64_t hard = min<int64_t>(soft * 4, tl * 3 / 4 - moveOverhead);
                hard = max<int64_t>(hard, 1);
                S.timed = true; S.softMs = max<int64_t>(1, min(soft, hard)); S.hardMs = hard;
            }
            // Lazy SMP: helper threads search the same position with a shared TT and staggered depths
            while ((int)g_helperPtrs.size() < g_threads - 1) {
                Searcher* h = new Searcher(); h->clearHistory(); h->threadId = (int)g_helperPtrs.size() + 1; g_helperPtrs.push_back(h);
            }
            S.threadId = 0; S.quiet = false;
            for (int i = 0; i < (int)g_helperPtrs.size() && i < g_threads - 1; i++) {
                Searcher* h = g_helperPtrs[i];
                h->pos = root; h->maxDepth = S.maxDepth; h->nodeLimit = 0; h->timed = S.timed; h->hardMs = S.hardMs; h->softMs = S.softMs;
                h->quiet = true; h->nodes = 0;
            }
            int nh = min((int)g_helperPtrs.size(), g_threads - 1);
            worker = thread([nh]() {
                g_helperStop = false;
                vector<thread> ht;
                for (int i = 0; i < nh; i++) { Searcher* h = g_helperPtrs[i]; ht.emplace_back([h]() { h->think(); }); }
                S.think();
                g_helperStop = true;
                for (auto& t : ht) t.join();
            });
        } else if (cmd == "stop") {
            g_stop = true; joinWorker(); g_stop = false;
        } else if (cmd == "quit") {
            g_stop = true; joinWorker();
            return 0;
        } else if (cmd == "perft") {
            int d = 1; is >> d;
            Move mv[256]; int n = legalMoves(root, mv); U64 total = 0;
            auto t = chrono::steady_clock::now();
            for (int i = 0; i < n; i++) {
                root.make(mv[i]); U64 c = d > 1 ? perft(root, d - 1) : 1; root.unmake();
                printf("%s: %llu\n", moveStr(mv[i]).c_str(), (unsigned long long)c); total += c;
            }
            auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - t).count();
            printf("\nNodes: %llu  (%lld ms)\n", (unsigned long long)total, (long long)ms); fflush(stdout);
        } else if (cmd == "status") {
            // debug/arbiter helper: game state of the current position
            Move mv[256]; int n = legalMoves(root, mv);
            int total = popcnt(root.all);
            bool insuff = !root.bb[0][PAWN] && !root.bb[1][PAWN] &&
                          !(root.bb[0][ROOK] | root.bb[1][ROOK] | root.bb[0][QUEEN] | root.bb[1][QUEEN]) && total <= 3;
            int reps = 0;
            for (int i = 2; i <= min(root.half, root.hp); i += 2) if (root.hist[root.hp - i].key == root.key) reps++;
            const char* st = "ongoing";
            if (n == 0) st = root.inCheck() ? "checkmate" : "stalemate";
            else if (insuff) st = "draw";
            else if (root.half >= 100) st = "draw";
            else if (reps >= 2) st = "draw";
            printf("status %s\n", st); fflush(stdout);
        } else if (cmd == "eval") {
            Pos q = root; q.track = false; int hce = evaluate(q);
            printf("hce %d cp", hce);
            if (NN.loaded) { q.track = true; q.refreshAcc(); printf("  nnue(%s) %d cp", g_netName.c_str(), evaluate(q)); }
            printf("  (side to move)\n"); fflush(stdout);
        } else if (cmd == "selftest") {
            nnueSelfTest();
        } else if (cmd == "legal") {
            Move mv[256]; int n = legalMoves(root, mv);
            printf("legal"); for (int i = 0; i < n; i++) printf(" %s", moveStr(mv[i]).c_str());
            printf("\n"); fflush(stdout);
        } else if (cmd == "datagen") {
            int games = 100, depth = 12, rp = 8; U64 nodes = 5000, seed = 1; string file = "data.txt";
            string tok;
            while (is >> tok) {
                if (tok == "games") is >> games; else if (tok == "nodes") is >> nodes; else if (tok == "depth") is >> depth;
                else if (tok == "seed") is >> seed; else if (tok == "file") is >> file; else if (tok == "randplies") is >> rp;
            }
            tt.resize(16);
            datagen(games, nodes, depth, file, seed, rp);
        } else if (cmd == "bench") {
            int d = 10; is >> d;
            S.quiet = true; U64 total = 0; auto tb = chrono::steady_clock::now();
            for (const char* f : BENCH_FENS) {
                Pos bp; bp.setFen(f); tt.clear(); S.clearHistory();
                S.pos = bp; S.maxDepth = d; S.nodeLimit = 0; S.timed = false; S.think();
                total += S.nodes;
            }
            S.quiet = false;
            auto ms = chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - tb).count();
            printf("bench nodes %llu time %lld ms nps %llu net %s\n", (unsigned long long)total, (long long)ms,
                   (unsigned long long)(total * 1000 / max<long long>(1, ms)), (NN.loaded && g_useNNUE) ? g_netName.c_str() : "hce");
            fflush(stdout);
        } else if (cmd == "d") {
            for (int r = 7; r >= 0; r--) {
                for (int f = 0; f < 8; f++) {
                    int sq = r * 8 + f, t = root.pt[sq];
                    char ch = '.';
                    if (t != NO_PIECE) { ch = "pnbrqk"[t]; if (root.occ[WHITE] & bit(sq)) ch = toupper(ch); }
                    printf("%c ", ch);
                }
                printf("\n");
            }
            printf("side: %s  key: %llx\n", root.stm == WHITE ? "white" : "black", (unsigned long long)root.key); fflush(stdout);
        }
    }
    joinWorker();
    return 0;
}
