#!/usr/bin/env python3
"""Materialize an ordered raw result containing exactly the chosen subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    raw_rows = [json.loads(line) for line in open(args.raw) if line.strip()]
    by_uid = {}
    for row in raw_rows:
        uid = row["uid"]
        if uid in by_uid:
            raise RuntimeError(f"duplicate raw uid: {uid}")
        by_uid[uid] = row
    subset_rows = [json.loads(line) for line in open(args.subset)
                   if line.strip()]
    ordered_uids = [row["uid"] for row in subset_rows]
    missing = [uid for uid in ordered_uids if uid not in by_uid]
    if missing:
        raise RuntimeError(f"missing {len(missing)} subset rows: {missing}")

    output = Path(args.output)
    with output.open("w") as handle:
        for uid in ordered_uids:
            handle.write(json.dumps(by_uid[uid], ensure_ascii=False) + "\n")
    print(f"wrote {output}: {len(ordered_uids)} turns")


if __name__ == "__main__":
    main()
