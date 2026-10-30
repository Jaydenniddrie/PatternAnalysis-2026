# Experiment Log

| ID | Date | Model | Data | Epochs | Batch | LR | Time | Peak VRAM | Hardware | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-001 | 2026-10-29 | DCGAN | train split (11,460 slices) | 30 | 64 | 2e-4 | 130.6 s | 0.27 GB (PyTorch-allocated) | A100 (Rangpur, job 714056) | D dominance increased over training (D(x) 0.83→0.95, D(G(z)) 0.19→0.05); G loss fell to ~ep 18 then rose. Samples show recognisable pelvic layout and bladder with good diversity (no visible mode collapse), but smeared tissue texture, missing/vague femoral heads, wavy body edges and a few washed-out samples. |


| EXP-002 | 2026-10-30 | WGAN-GP (n_critic 5, λ 10) | train split (11,460 slices) | 30 | 64 | 1e-4, betas (0, 0.9) | 660.8 s | 0.34 GB (PyTorch-allocated) | A100 (Rangpur, job 715510) | Stable training: no loss spikes, GP steady ~0.18–0.31. Wasserstein estimate dipped to 8.6 then rose and plateaued ~14.7 (ambiguous: critic sharpening vs G stalling). ~10× slower per epoch than DCGAN (19.5 s vs 1.9 s). Samples: same layout and diversity as DCGAN, no mode collapse; softer/blurrier with horizontal streak artefacts instead of DCGAN's swirls; femoral heads still mostly missing. Not visibly better than DCGAN by eye. |
