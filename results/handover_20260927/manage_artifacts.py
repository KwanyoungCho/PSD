"""Inventory/verify this handoff and package raw research data outside Git.

Uses only the standard library and never touches models or GPUs. Run from any
directory. Manifests are repository-relative and preserve old frozen outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROOTS = [
    "results/residial_dist", "results/duet_calibration",
    "results/duet_tree_analysis", "results/duet_tree_al_full",
    "results/duet_tree_posthoc", "results/duet_tree_followup",
    "ssd/experiments/proxy_source_ablation",
]
MANIFEST = HERE / "artifact_manifest.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_file(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or "\n" in name or "\0" in name:
        raise ValueError(f"Unsafe manifest path: {name!r}")
    return ROOT / path


def classify(path):
    if path.suffix == ".npz":
        return "archive", "full distribution / diagnostic chunk"
    if path.name.startswith("duet_profile_") and path.suffix == ".json":
        return "archive", "raw timing profile"
    if path.stat().st_size > 10 * 1024**2:
        return "archive", "large per-tree / derived record"
    return "git", "source / report / plan / compact evidence"


def inventory():
    entries = []
    omitted = []
    for name in ROOTS:
        for path in sorted((ROOT / name).rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo", ".lock", ".tmp"}:
                omitted.append(rel)
                continue
            if path.is_symlink():
                raise ValueError(f"Review symlink explicitly: {rel}")
            group, reason = classify(path)
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"File changed while reading: {rel}")
            entries.append(dict(path=rel, bytes=after.st_size, sha256=digest,
                                group=group, reason=reason))
    summary = {}
    for group in ("git", "archive"):
        chosen = [e for e in entries if e["group"] == group]
        summary[group] = dict(files=len(chosen), bytes=sum(e["bytes"] for e in chosen))
        (HERE / f"{group}_files.txt").write_text(
            "".join(e["path"] + "\n" for e in chosen))
        sums_name = "raw_SHA256SUMS" if group == "archive" else "git_SHA256SUMS"
        (HERE / sums_name).write_text(
            "".join(e["sha256"] + "  " + e["path"] + "\n" for e in chosen))
    manifest = dict(schema="duet_handoff_artifacts_v1",
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    base_commit="a82f7d24fb36827a9a81a3567f344dccb71f193e",
                    roots=ROOTS, summary=summary, files=entries,
                    omitted_regenerable=omitted)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def verify(group):
    manifest = json.loads(MANIFEST.read_text())
    errors = []
    counts = Counter()
    for entry in manifest["files"]:
        if group != "all" and entry["group"] != group:
            continue
        path = relative_file(entry["path"])
        if not path.is_file():
            errors.append(f"Missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            errors.append(f"Changed: {entry['path']}")
        counts[entry["group"]] += 1
    print(json.dumps(dict(group=group, checked=dict(counts), errors=errors), indent=2))
    if errors:
        raise SystemExit(1)


def package(output):
    manifest = json.loads(MANIFEST.read_text())
    output = output.resolve()
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise ValueError("Refusing to overwrite an existing archive/partial")
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = [e for e in manifest["files"] if e["group"] == "archive"]
    for entry in entries:
        path = relative_file(entry["path"])
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing/changed input: {path}")
    # Names are passed as NUL-delimited data, not interpreted as shell code.
    names = b"".join(e["path"].encode() + b"\0" for e in entries)
    partial = output.with_suffix(output.suffix + ".partial")
    subprocess.run(["tar", "--create", "--file", str(partial), "--directory", str(ROOT),
                    "--null", "--verbatim-files-from", "--files-from", "-"],
                   input=names, check=True)
    partial.rename(output)
    digest = sha(output)
    summary = dict(archive_basename=output.name, bytes=output.stat().st_size,
                   sha256=digest, files=len(entries),
                   uncompressed_file_bytes=sum(e["bytes"] for e in entries),
                   artifact_manifest_sha256=sha(MANIFEST),
                   created_utc=datetime.now(timezone.utc).isoformat())
    (HERE / "archive_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (HERE / "archive_SHA256SUMS").write_text(f"{digest}  {output.name}\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--group", choices=["git", "archive", "all"], default="all")
    pack_parser = sub.add_parser("package")
    pack_parser.add_argument("--output", type=Path,
        default=ROOT / "handoff_artifacts/duet_research_raw_20260927.tar")
    args = parser.parse_args()
    if args.command == "inventory":
        inventory()
    elif args.command == "verify":
        verify(args.group)
    else:
        package(args.output)


if __name__ == "__main__":
    main()
