"""Compare a regenerated results directory with the committed results/.

    BCP_RESULTS=results_regen python run_all.py --from 2
    python check_reproduction.py results_regen

Every CSV in results/tables and results/source_data is compared cell by cell: numbers with a relative tolerance
of 1e-9, text exactly. Figures and workbooks are not compared, because they are drawn from these CSVs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
new_root = Path(sys.argv[1]) if len(sys.argv) > 1 else sys.exit(__doc__)
ref_root = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO / 'results'


def same(a, b):
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return f'shape or columns differ: {a.shape} vs {b.shape}'
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            if not np.allclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True):
                return f'column {c}: max abs difference {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.3g}'
        elif not x.fillna('<NA>').astype(str).equals(y.fillna('<NA>').astype(str)):
            return f'column {c}: text differs'
    return ''


failures = checked = 0
for sub in ['tables', 'source_data']:
    for ref in sorted((ref_root / sub).glob('*.csv')):
        new = new_root / sub / ref.name
        if not new.exists():
            print(f'MISSING  {sub}/{ref.name}'); failures += 1; continue
        problem = same(pd.read_csv(ref), pd.read_csv(new)); checked += 1
        print(f"{'DIFFERS' if problem else 'same   '}  {sub}/{ref.name}  {problem}")
        failures += bool(problem)
print(f'\n{checked} files compared, {failures} with differences')
sys.exit(1 if failures else 0)
