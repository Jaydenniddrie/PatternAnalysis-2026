# Experiment Log

| ID | Date | Model | Data | Epochs | Batch | LR | Time | Peak VRAM | Hardware | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-10-29 | DCGAN | train split (11,460 slices) | 30 | 64 | 2e-4 | 130.6 s | 0.27 GB (PyTorch-allocated) | A100 (Rangpur, job 714056) | D dominance increased over training (D(x) 0.83→0.95, D(G(z)) 0.19→0.05); G loss fell to ~ep 18 then rose. Samples show recognisable pelvic layout and bladder with good diversity (no visible mode collapse), but smeared tissue texture, missing/vague femoral heads, wavy body edges and a few washed-out samples. |
