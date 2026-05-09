"""Astabench override for inspect_ai's `react` solver.

Stock `inspect_ai.agent.react` only sees tools passed via its own `tools=`
kwarg.  Tools registered by the task's `setup=[use_tools(...)]` step land on
`state.tools` and are silently dropped when CLI invokes `--solver react`,
because Inspect-AI's `as_solver()` only forwards `state.messages` into the
inner `AgentState` (see `inspect_ai/agent/_as_solver.py`).

This module registers a solver also named `react` that:

  1. Captures `state.tools` (typically populated by the task's `use_tools(...)`).
  2. Captures any explicit `tools=` kwargs (e.g. from `-S tools=...`, rare).
  3. Calls `inspect_ai.agent.react(tools=<combined>, **rest_kwargs)`.
  4. Bridges that agent into a Solver via `as_solver()`.

Inspect-AI resolves `--solver react` against the solver registry before
falling back to the agent registry, so this registration shadows the
built-in transparently — invocations like
`uv run inspect eval astabench/super_test --solver react ...` keep working
verbatim and the agent now actually sees `python_session` (and any other
task-provided tool).

Forwarded kwargs include all of `react()`'s knobs:
`name, description, prompt, model, attempts, submit, on_continue,
retry_refusals, compaction, truncation, approval`.
"""

from inspect_ai.agent import as_solver
from inspect_ai.agent import react as _builtin_react
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def react(**kwargs) -> Solver:
    """ReAct solver that respects task-provided tools (`state.tools`)."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        kwarg_tools = list(kwargs.pop("tools", None) or [])
        task_tools = list(state.tools or [])
        # Task tools first (they may carry date/id restrictions); kwarg-side
        # tools appended only if not already present by identity.
        all_tools = task_tools + [t for t in kwarg_tools if t not in task_tools]
        agent = _builtin_react(tools=all_tools, **kwargs)
        return await as_solver(agent)(state, generate)

    return solve
