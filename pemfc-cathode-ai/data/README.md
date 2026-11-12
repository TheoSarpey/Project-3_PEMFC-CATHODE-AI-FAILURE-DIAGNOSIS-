# Data

| Folder | What | In git? |
|---|---|---|
| `raw/` | ECSIM PEM fuel-cell dataset (polarisation + impedance), downloaded by `python download_data.py` | **No** (git-ignored) |
| `knowledge/` | 30 short original notes (`K01`..`K30`) used by the RAG system. Each has an `evidence` label: `textbook`, `literature`, `dataset` | Yes (small, hand-written) |
| `reviews/` | `user_reviews.jsonl` written by the app when you save comments | **No** (git-ignored, may contain personal text) |
| `literature_watchlist_template.csv` | Header-only template for the R&D monitor (`doi,title,authors,year,journal,abstract`) | Yes |

## Source and licence of the raw data
Hamidi S., Haghighi S., Askari K. *Dataset of Standard Tests of Nafion 112 Membrane and Membrane Electrode
Assembly (MEA) Activation Tests of Proton Exchange Membrane (PEM) Fuel Cell.* ChemRxiv (2020).
doi:10.26434/chemrxiv.11902023 · https://github.com/ECSIM/pem-dataset1 · **CC BY 4.0** (keep the attribution).

Content: Pt/C (20 wt%) cathodes and anodes on carbon paper, Nafion 112, H2/O2, 5/15/25 psig, 30/50/80/100 % RH,
three activation protocols, EIS without a frequency column.

## Traps
* `power_density` = current x voltage: never a predictor (leakage).
* Pressure/RH are missing for the activation files that are indexed by `set` (missing = structure, not noise).
* Compression and Nafion % exist only in the Nafion standard test (file `1.csv`).
* One catalyst type, oxygen not air, one laboratory: no conclusion about other catalysts or air operation.

## Knowledge notes
Written for this project in original wording from standard electrochemistry. They contain **no DOIs, paper
titles or catalyst performance numbers** except the dataset citation. Notes labelled `literature` describe
commonly reported behaviour whose magnitude depends on the study: verify before citing.
