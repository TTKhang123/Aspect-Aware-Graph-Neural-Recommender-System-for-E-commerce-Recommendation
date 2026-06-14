from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    path = Path(args.path)
    pf = pq.ParquetFile(path)
    print(f"File: {path}")
    print("Columns:")
    for name in pf.schema.names:
        print(f"- {name}")


if __name__ == "__main__":
    main()
