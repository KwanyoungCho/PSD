"""Run coverage jobs sequentially on one GPU pair, retaining telemetry.

Plans are JSON lists of {name, args: [CLI args], env: {optional overrides}}.
Each job starts a fresh engine, including the draft RNG. Existing completed
results may be resumed; failed results require a new output directory.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def query(args):
    return subprocess.check_output(["nvidia-smi", *args], text=True).strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--gpus", default="2,3")
    p.add_argument("--port", type=int, default=24500)
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    a.directory = a.directory.resolve()
    a.directory.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(a.plan.read_text())
    (a.directory / "plan.json").write_text(json.dumps(jobs, indent=2))
    uuids = set(query(["-i", a.gpus, "--query-gpu=uuid", "--format=csv,noheader"]).splitlines())

    def processes():
        rows = query(["--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                      "--format=csv,noheader,nounits"]).splitlines()
        return [r for r in rows if r.split(",")[0] in uuids]

    manifest = []
    for i, job in enumerate(jobs):
        name = job["name"]
        if Path(name).name != name:
            raise ValueError("Job names must be simple filenames")
        output = a.directory / (name + ".json")
        if output.exists():
            if json.loads(output.read_text()).get("status") == "complete":
                print("SKIP", name, flush=True)
                continue
            raise RuntimeError(f"Incomplete output exists: {output}")
        busy = processes()
        if busy:
            raise RuntimeError(f"Selected GPUs already busy; no job started: {busy}")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpus, SSD_CUDA_ARCH="8.9",
                   SSD_ATTN_BACKEND="auto", SSD_DIST_PORT=str(a.port+i),
                   OMP_NUM_THREADS="4", SSD_SEED="0", SSD_CHAIN_PROXY_GRAPH="1",
                   SSD_BATCHED_PROXY_GRAPH="1", SSD_DUET_EXIT_REPLICA="0",
                   SSD_ASYNC_PROXY_SEND="1", SSD_PROXY_STREAM="0")
        env.update(job.get("env", {}))
        cmd = [sys.executable, "-O", str(Path(__file__).with_name("mlsys_coverage.py")),
               *job["args"], "--output", str(output)]
        row = dict(name=name, command=cmd, started=time.time(), gpu_uuids=sorted(uuids),
                   external_pids=[], samples=0)
        telemetry = a.directory / (name + ".telemetry.jsonl")
        print("START", name, flush=True)
        with (a.directory / (name + ".log")).open("w") as log, telemetry.open("w") as trace:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            while proc.poll() is None:
                active = processes()
                external = []
                for entry in active:
                    pid = int(entry.split(",")[1])
                    try:
                        if os.getpgid(pid) != proc.pid:
                            external.append(pid)
                    except ProcessLookupError:
                        pass
                row["external_pids"] = sorted(set(row["external_pids"] + external))
                sample = dict(time=time.time(), processes=active, external_pids=external,
                              gpu=query(["-i", a.gpus,
                                  "--query-gpu=index,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu",
                                  "--format=csv,noheader,nounits"]))
                trace.write(json.dumps(sample) + "\n")
                trace.flush()
                row["samples"] += 1
                if time.time() - row["started"] > a.timeout:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                    row["timeout"] = True
                    break
                time.sleep(2)
            proc.wait()
        row.update(ended=time.time(), returncode=proc.returncode)
        result = json.loads(output.read_text()) if output.exists() else {}
        row["status"] = result.get("status", "missing")
        manifest.append(row)
        (a.directory / "campaign.json").write_text(json.dumps(manifest, indent=2))
        print("END", name, row["status"], "external", row["external_pids"], flush=True)
        if row["status"] != "complete" or row["returncode"] != 0:
            raise RuntimeError(f"Failed job: {name}; inspect its log")
        # Workers normally have exited already. Never kill unrelated processes.
        for _ in range(10):
            if not processes():
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
