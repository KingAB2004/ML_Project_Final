# Data-quality evaluation: our corpus next to ExTES and ESConv

Shape of the baseline paper's Tables 2, 6 and 8 (COCOON, EMNLP 2025). Our rows and the ExTES / ESConv rows marked *ours* are measured here with the same code and scorer; *paper* rows are copied from the paper for reference (their judge is GPT-4o and some of their data is Chinese). Script: `extra/data_quality.py`.

## User-description diversity (paper Table 8)

1000 descriptions per dataset (sampled, seed 0); self-BLEU of 300 descriptions against 100 others each. BLEU lower = more diverse; D-2 and entropy higher = more diverse.

| Dataset | source | BLEU-2 ↓ | BLEU-4 ↓ | D-2 ↑ | Shannon entropy ↑ | mean words |
|---|---|---|---|---|---|---|
| **ours (this corpus)** | ours | 0.702 | 0.398 | 0.256 | 8.38 | 106 |
| ExTES | ours | 0.724 | 0.450 | 0.237 | 7.80 | 26 |
| ESConv | ours | 0.498 | 0.146 | 0.536 | 8.52 | 23 |
| COCOON | paper | 0.760 | 0.287 | 0.476 | 10.17 | - |
| extes | paper | 0.875 | 0.678 | 0.323 | 7.89 | - |
| esconv | paper | 0.747 | 0.540 | 0.508 | 8.24 | - |

## Dialogue quality, judge scales (paper Table 2)

Six basic metrics on 0-100, CRS Aff / Neg and RAC Sup / Man on 1-7 (Neg lower is better). Our judge: Mistral-Nemo-12B (paper: GPT-4o).

| Dataset | source | n | Flu | Div | Emp | Inf | Hum | Skil | Basic Avg | Aff | Neg ↓ | Sup | Man |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **ours (this corpus)** | ours | 200 | 83.3 | 67.2 | 83.3 | 83.2 | 83.3 | 83.0 | 80.5 | 5.72 | 1.00 | 6.26 | 5.57 |
| ExTES | ours | 100 | 83.3 | 73.2 | 84.2 | 83.3 | 83.3 | 83.3 | 81.8 | 5.98 | 1.00 | 6.74 | 6.07 |
| ESConv | ours | 100 | 82.7 | 66.0 | 79.0 | 81.2 | 83.0 | 81.5 | 78.9 | 5.88 | 1.00 | 6.28 | 5.63 |
| esconv | paper | - | 76.0 | 65.9 | 75.9 | 62.1 | 71.1 | 69.6 | 70.1 | 5.12 | 1.74 | 6.03 | 5.22 |
| extes | paper | - | 91.8 | 81.3 | 92.5 | 81.1 | 89.2 | 90.3 | 87.7 | 5.56 | 1.48 | 6.79 | 6.03 |
| COCOON | paper | - | 93.1 | 83.1 | 95.7 | 86.1 | 89.5 | 93.3 | 90.1 | 5.84 | 1.21 | 6.90 | 6.20 |

## Dialogue quality, ESC-RANK (paper Table 6)

ESC-RANK scores each dialogue 0-4 per dimension; shown x 25 as in the paper (its ceiling 75 = every dialogue at 3). Same scorer as the paper, so these rows are the most directly comparable. Run 18-20 Oct on 10.10.10.104 (`scripts/escrank_score.py` in its own venv: transformers 4.41, sentencepiece 0.2.0, protobuf); Inf answers longer than the 8-token budget were re-asked with 16 tokens (`--fill-missing`). Hum sits near 50 for all three corpora, far below the paper's ExTES / COCOON rows.

| Dataset | source | n | Flu | Div | Emp | Inf | Hum | Skil |
|---|---|---|---|---|---|---|---|---|
| **ours (this corpus)** | ours | 200 | 75.0 | 64.9 | 75.0 | 56.9 | 49.9 | 65.1 |
| ExTES | ours | 100 | 75.0 | 74.0 | 75.0 | 70.2 | 49.2 | 74.0 |
| ESConv | ours | 100 | 71.0 | 54.8 | 74.8 | 54.5 | 50.2 | 67.5 |
| esconv | paper | - | 72.3 | 55.8 | 74.0 | 56.5 | 50.8 | 69.5 |
| extes | paper | - | 75.0 | 75.0 | 75.0 | 75.0 | 70.3 | 74.8 |
| COCOON | paper | - | 75.0 | 75.0 | 75.0 | 75.0 | 75.0 | 75.0 |
