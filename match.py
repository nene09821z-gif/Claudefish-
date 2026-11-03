#!/usr/bin/env python3
# Minimal UCI match runner. Uses the ClaudeFish "status"/"legal" helper commands as arbiter.
# usage: match.py engineA engineB games movetime_ms [optsA] [optsB]   (opts: "Name=Value;Name=Value")
import subprocess, sys, random, time

class Eng:
    def __init__(self, path, opts):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci"); self.wait("uciok")
        for kv in filter(None, opts.split(";")):
            k, v = kv.split("=", 1); self.send(f"setoption name {k} value {v}")
        self.send("isready"); self.wait("readyok")
    def send(self, s): self.p.stdin.write(s + "\n"); self.p.stdin.flush()
    def wait(self, tok):
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("engine died")
            if l.startswith(tok): return l.strip()
    def query(self, cmd, tok):
        self.send(cmd); return self.wait(tok)
    def close(self):
        try: self.send("quit")
        except Exception: pass

def play(white, black, opening, mt):
    moves = list(opening)
    for e in (white, black): e.send("ucinewgame"); e.send("isready"); e.wait("readyok")
    while True:
        pos = "position startpos moves " + " ".join(moves) if moves else "position startpos"
        white.send(pos)
        st = white.query("status", "status").split()[1]
        if st == "checkmate": return "black" if len(moves) % 2 == 0 else "white"
        if st in ("stalemate", "draw"): return "draw"
        if len(moves) > 400: return "draw"
        eng = white if len(moves) % 2 == 0 else black
        eng.send(pos); eng.send(f"go movetime {mt}")
        bm = eng.wait("bestmove").split()[1]
        if bm == "0000": return "draw"
        moves.append(bm)

def random_opening(e, plies, rng):
    moves = []
    for _ in range(plies):
        e.send("position startpos moves " + " ".join(moves) if moves else "position startpos")
        leg = e.query("legal", "legal").split()[1:]
        if not leg: break
        moves.append(rng.choice(leg))
    return moves

def main():
    a, b, games, mt = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    oa = sys.argv[5] if len(sys.argv) > 5 else ""; ob = sys.argv[6] if len(sys.argv) > 6 else ""
    A, B = Eng(a, oa), Eng(b, ob)
    rng = random.Random(int(time.time()) % 1000 if len(sys.argv) <= 7 else int(sys.argv[7]))
    w = d = l = 0
    for g in range(games // 2):
        op = random_opening(A, 6, rng)
        for first in (0, 1):
            white, black = (A, B) if first == 0 else (B, A)
            r = play(white, black, op, mt)
            if r == "draw": d += 1
            elif (r == "white") == (first == 0): w += 1
            else: l += 1
        n = w + d + l
        score = (w + d / 2) / n
        import math
        elo = -400 * math.log10(1 / max(score, 1e-3) - 1) if 0 < score < 1 else (999 if score >= 1 else -999)
        print(f"after {n}: +{w} ={d} -{l}  score {score:.3f}  elo {elo:+.0f}", flush=True)
    A.close(); B.close()
main()
