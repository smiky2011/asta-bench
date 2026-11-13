#!/usr/bin/env python3
"""Original-benchmark side of the Harbor <-> AstaBench parity experiment.

The Harbor adapter (harbor-framework/adapters, `src/asta-bench`) generates one
Harbor task per AstaBench sample.  In its Inspect-AI bridge packaging every
task carries `environment/workspace/inspect_config.json`, which names the
Inspect task, the sample id and the task arguments the Harbor side runs.  This
script reads those files and runs the *same* samples here, with this
repository's own scorers, so both sides evaluate identical samples with an
identical solver.

    # 1. In the adapters repo: generate the parity subset
    #    (cd src/asta-bench && uv run asta-bench --split parity --output-dir /tmp/parity-tasks)
    # 2. Here (Python 3.11 environment, `uv sync` done):
    export ANTHROPIC_API_KEY=... HF_TOKEN=... ASTA_TOOL_KEY=...
    uv run python scripts/harbor_parity_original_side.py \
        --tasks-dir /tmp/parity-tasks --log-dir logs/harbor-parity/run-1

One `inspect eval` is started per sample, each with its own log directory, so
the run is resumable (samples that already have a log are skipped) and one
failing sample cannot affect another.  Credentials are read from the
environment; nothing is written to this file.

What is the same as the Harbor side: task, sample id, `-T` task args, the
solver (`astabench/react`), the model and `--max-tool-output`.  What differs
by design: the Harbor bridge passes `--no-score` (the Harbor verifier scores)
and, for code-execution tasks, `--sandbox local` (the solver's tools run in
the Harbor task container); here the task's own Docker sandbox and scorer are
used, so those two flags are dropped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_SOLVER = "astabench/react"
DEFAULT_MODEL = "anthropic/claude-haiku-4-5-20251001"

# paper_finder keeps a run-mutated cache of relevance judgements next to its
# code.  It is not part of the benchmark and the Harbor verifier has no such
# cache, so a parity run must start without it.
_PAPER_FINDER_CACHE = (
    Path(__file__).resolve().parent.parent
    / "astabench"
    / "evals"
    / "paper_finder"
    / "detailed_reference.json"
)


def _strip_bridge_only_args(eval_args: list[str]) -> list[str]:
    """Drop `--sandbox <x>` (Harbor-bridge only); keep everything else."""
    out: list[str] = []
    skip = False
    for arg in eval_args:
        if skip:
            skip = False
            continue
        if arg == "--sandbox":
            skip = True
            continue
        if arg.startswith("--sandbox=") or arg == "--no-score":
            continue
        out.append(arg)
    return out


def _load_tasks(tasks_dir: Path) -> list[tuple[str, dict]]:
    tasks = []
    for cfg_path in sorted(tasks_dir.glob("*/environment/workspace/inspect_config.json")):
        task_id = cfg_path.parents[2].name
        tasks.append((task_id, json.loads(cfg_path.read_text(encoding="utf-8"))))
    return tasks


def _command(cfg: dict, log_dir: Path, solver: str, model: str) -> list[str]:
    cmd = [
        "inspect",
        "eval",
        cfg["task"],
        "--solver",
        solver,
        "--model",
        model,
        "--sample-id",
        str(cfg["sample_id"]),
        "--log-format",
        "json",
        "--log-dir",
        str(log_dir),
    ]
    for key, value in (cfg.get("task_args") or {}).items():
        cmd += ["-T", f"{key}={value}"]
    for key, value in (cfg.get("solver_args") or {}).items():
        cmd += ["-S", f"{key}={value}"]
    cmd += _strip_bridge_only_args(list(cfg.get("eval_args") or []))
    return cmd


def _run_one(task_id: str, cfg: dict, root: Path, solver: str, model: str) -> str:
    log_dir = root / task_id
    if any(log_dir.glob("*.json")):
        return f"[skip] {task_id} (log exists)"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = _command(cfg, log_dir, solver, model)
    (log_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    with (log_dir / "stdout.log").open("w", encoding="utf-8") as out:
        proc = subprocess.run(cmd, check=False, stdout=out, stderr=subprocess.STDOUT)
    status = "done" if proc.returncode == 0 else f"FAIL rc={proc.returncode}"
    return f"[{status}] {task_id}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--solver", default=DEFAULT_SOLVER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument(
        "--families",
        nargs="+",
        default=None,
        help="Only tasks whose Harbor task id starts with one of these prefixes",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the commands and exit"
    )
    args = parser.parse_args()

    tasks = _load_tasks(args.tasks_dir)
    if args.families:
        tasks = [t for t in tasks if any(t[0].startswith(f + "-") for f in args.families)]
    if not tasks:
        print(
            f"No inspect_config.json found under {args.tasks_dir} "
            "(generate the Inspect-AI bridge packaging, not --cli)",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        for task_id, cfg in tasks:
            cmd = _command(cfg, args.log_dir / task_id, args.solver, args.model)
            print(" ".join(cmd))
        return 0

    if _PAPER_FINDER_CACHE.exists():
        print(
            f"Refusing to start: {_PAPER_FINDER_CACHE} exists. It is a local, "
            "run-mutated cache of paper_finder relevance judgements that would "
            "make this side score differently from a clean checkout. Move or "
            "delete it, then re-run.",
            file=sys.stderr,
        )
        return 2

    args.log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [
            pool.submit(_run_one, task_id, cfg, args.log_dir, args.solver, args.model)
            for task_id, cfg in tasks
        ]
        for fut in futures:
            line = fut.result()
            failures += line.startswith("[FAIL")
            print(line, flush=True)
    print(f"{len(tasks)} samples, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
