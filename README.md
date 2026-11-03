# ClaudeFish 3.0

Build: `g++ -O3 -march=native -std=c++17 -pthread -o claudefish claudefish3.cpp`
(on phone/Termux or CI drop `-march=native`)

## Trạng thái thật (đã test)
- NNUE đã có đủ: 768 -> 2x128 SCReLU -> 1, 8 output bucket, int16, cập nhật tăng dần. `selftest` PASSED, perft 5 đúng.
- Net thử `trial.nnue` (huấn luyện 268k vị trí, ~15s) **yếu hơn eval tay khoảng -400 Elo** (40 ván) => NNUE **tắt mặc định**.
  Bật thử: `setoption name EvalFile value trial.nnue` + `setoption name UseNNUE value true`.
- Lazy SMP: `setoption name Threads value N` (chỉ chạy thử đa luồng, chưa đo tăng tốc vì máy test 1 nhân).
- ProbCut / correction history / cut-node (PC, CORR, CUT) là tùy chọn thử nghiệm, tắt: bản bật chúng thua v2 trong 24 ván.
- Bản mặc định ngang v2 (+10 =8 -12 trên 30 ván, nhiễu lớn): chưa chứng minh mạnh hơn.

## Làm NNUE mạnh thật
Cần hàng chục triệu vị trí. Chạy workflow `Train NNUE` trên GitHub Actions (hoặc PC):
`datagen games N nodes 5000 file data.txt` -> `./cf_train --data data.txt --out claudefish.nnue`
-> `python3 net2h.py claudefish.nnue > cf_net.h` (nhúng vào engine) -> so sánh bằng `match.py`.
Lệnh phụ: `bench`, `selftest`, `eval`, `legal`, `datagen`.
