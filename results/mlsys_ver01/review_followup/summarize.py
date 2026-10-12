"""Summarize review-only regression and diagnostic runs without replacing the 480-input evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

R = Path(__file__).resolve().parent
ROOT = R.parents[2]
spec = importlib.util.spec_from_file_location("review_components", R.parent / "analyze_components.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.R = R

components = {name: module.analyze(R / "profile_runs" / (name + ".json"))
              for name in ("before_profile", "after_profile")}
for arm in components.values():
    # Per-step timelines remain reproducible from the archived raw events.
    # This review needs interval statistics rather than another large timeline.
    arm.pop("timeline")
(R / "COMPONENTS.json").write_text(json.dumps(dict(
    scope="One 24-input diagnostic pair, 4b0d208 versus follow-up runtime. Not an uninstrumented throughput experiment.",
    arms=components), indent=2) + "\n")
before, after = [json.loads((R / "profile_runs" / (name + ".json")).read_text())
                 for name in ("before_profile", "after_profile")]
assert before["status"] == after["status"] == "complete"
summary = dict(profile=dict(outputs_equal=before["outputs"] == after["outputs"],
               inputs=len(before["outputs"]), before=before["summary"], after=after["summary"]),
               fallback=[], components={})
for key in ("target_pre_ms", "target_post_ms", "target_tree_walk_ms",
            "draft_p1_replay_ms", "draft_p2_replay_ms", "draft_p1_total_ms", "draft_p2_total_ms"):
    summary["components"][key] = {name: arm["summary"][key]
                                   for name, arm in components.items()}
for name in ("arena_exec_off", "arena_executor_none", "cpu_fallback"):
    raw = json.loads((R / "fallback" / (name + ".json")).read_text())
    assert raw["status"] == "complete"
    trace = [json.loads(line) for line in (R / "fallback" / (name + ".trace.jsonl")).read_text().splitlines()]
    assert trace and all(t["first_child_argmax"] for t in trace)
    summary["fallback"].append(dict(name=name, inputs=len(raw["outputs"]),
         calls=len(trace), rows=sum(t["rows"] for t in trace),
         callers=sorted(set(t["caller"] for t in trace)), all_first_child_argmax=True,
         scope="Assertions include warmup; diagnostic synchronization excludes TPS use."))
campaigns = [json.loads((R / folder / "campaign.json").read_text())
             for folder in ("fallback", "profile_runs")]
summary["jobs_complete"] = all(j["status"] == "complete" and j["exit_code"] == 0
                                for campaign in campaigns for j in campaign)
summary["external_gpu_interference"] = any(j["external_pids"] for campaign in campaigns for j in campaign)
assert summary["jobs_complete"] and not summary["external_gpu_interference"]
(R / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")

# Preserve the original RUNTIME.patch as measured. This is a separate current,
# cumulative patch, checked against fresh paper files rather than a dirty tree.
patch = subprocess.check_output(["git", "diff", "a82f7d2", "--", "ssd/ssd", "ssd/bench/bench.py"], cwd=ROOT)
(R / "RUNTIME.patch").write_bytes(patch)
files = subprocess.check_output(["git", "diff", "--name-only", "a82f7d2", "--", "ssd/ssd", "ssd/bench/bench.py"], cwd=ROOT, text=True).splitlines()
with tempfile.TemporaryDirectory(prefix="ver01-review-patch-") as temp:
    temp = Path(temp)
    for file in files:
        result = subprocess.run(["git", "show", "a82f7d2:" + file], cwd=ROOT, capture_output=True)
        if result.returncode == 0:
            (temp / file).parent.mkdir(parents=True, exist_ok=True)
            (temp / file).write_bytes(result.stdout)
    subprocess.run(["git", "apply", "--check", str(R / "RUNTIME.patch")], cwd=temp, check=True)
    subprocess.run(["git", "apply", str(R / "RUNTIME.patch")], cwd=temp, check=True)
    assert all((temp / f).read_bytes() == (ROOT / f).read_bytes() for f in files)
(R / "PATCH_CHECK.json").write_text(json.dumps(dict(base="a82f7d2",
    applied_to_fresh_original_files=True, all_resulting_files_equal_current=True,
    files={f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}), indent=2) + "\n")

raw_files = sorted(p for folder in ("fallback", "profile_runs", "profiles")
                   for p in (R / folder).rglob("*") if p.is_file())
archive = R / "RAW_RESULTS.tar.gz"
with tarfile.open(archive, "w:gz") as tf:
    for p in raw_files:
        tf.add(p, arcname=str(p.relative_to(R)))
manifest = {str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_files}
with tarfile.open(archive) as tf:
    for member in tf.getmembers():
        assert hashlib.sha256(tf.extractfile(member).read()).hexdigest() == manifest[member.name]
(R / "RAW_MANIFEST.json").write_text(json.dumps(dict(files=manifest,
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), readback_verified=True), indent=2) + "\n")
print(json.dumps(summary, indent=2))
