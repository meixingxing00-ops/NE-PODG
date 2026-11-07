"""results/<文件>.csv 按区段取中位数（%）。用法：python -m cases.nc.explore.summary 文件 列1 列2 ..."""
import csv
import sys
from pathlib import Path

from cases.nc.explore.common import seg_table

rows = list(csv.DictReader(open(Path(__file__).resolve().parent / "results" / f"{sys.argv[1]}.csv", encoding="utf-8")))
cols = sys.argv[2:]
for r in rows:
    for c in cols:
        r[c] = float(r[c])
print("| 区段 | n | " + " | ".join(cols) + " |")
print(seg_table(rows, cols))
