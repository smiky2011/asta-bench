"""Astabench solver overrides.

Imports here run when `astabench.evals._registry` is loaded (the inspect_ai
entry-point), which registers the solvers in the inspect-ai registry.  The
local `react` solver shadows the inspect-ai built-in so `--solver react`
transparently picks up task-provided tools from `state.tools`.
"""

from astabench.solvers.react import react

__all__ = ["react"]
