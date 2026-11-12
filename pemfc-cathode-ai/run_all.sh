#!/usr/bin/env bash
# Recreate everything from a fresh clone: data -> notebooks (with outputs) -> tests.
set -euo pipefail
python download_data.py
for nb in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 "$nb"
done
pytest -q
echo "Now run: streamlit run app/streamlit_app.py"
