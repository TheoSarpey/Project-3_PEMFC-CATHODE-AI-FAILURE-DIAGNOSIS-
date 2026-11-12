"""Download the ECSIM PEM fuel-cell dataset into data/raw/ (never committed).

    python download_data.py                  # from GitHub (release v1.1)
    python download_data.py --from-dir PATH  # copy an existing copy (e.g. Project 2's data/raw)

Source : Hamidi S., Haghighi S., Askari K. (2020), ChemRxiv, doi:10.26434/chemrxiv.11902023
Repo   : https://github.com/ECSIM/pem-dataset1   Licence: CC BY 4.0 (keep the attribution)
The Streamlit app does the same thing from its first-run screen.
"""
import argparse
from pathlib import Path

from src.bootstrap import copy_ecsim, download_ecsim, raw_data_ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path)
    a = ap.parse_args()
    if a.from_dir:
        n = copy_ecsim(a.from_dir)
    elif raw_data_ok():
        print("data/raw already populated, skipping")
        return
    else:
        n = download_ecsim()
    print(f"done: {n} CSV files in data/raw")


if __name__ == "__main__":
    main()
