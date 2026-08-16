"""`python -m career_model <stage>` -- the whole pipeline from one entry point.

Having a single entry point matters for more than convenience: running
`python -m career_model.pipeline` directly would pickle the fitted model with
its classes rooted at `__main__`, which no later process can resolve.
"""

from __future__ import annotations

import sys

STAGES = {
    "panel": ("career_model.data.build_panel", "main"),
    "fit": ("career_model.pipeline", "main"),
    "posterior": ("career_model.fit_posterior", "main"),
    "diagnostics": ("career_model.diagnostics", "main"),
    "precompute": ("career_model.simulate.precompute", "main"),
    "backtest": ("career_model.validate.backtest", "main"),
    "backtest_ui": ("career_model.validate.backtest_ui", "main"),
    "compare": ("career_model.validate.compare_lgbm", "main"),
    "freeze_baseline": ("career_model.validate.freeze_baseline", "main"),
    "diagnose_projection": ("career_model.diagnose_projection", "main"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print("usage: python -m career_model <stage> [options]\n\nstages:")
        for k, (mod, _) in STAGES.items():
            print(f"  {k:<12s} {mod}")
        raise SystemExit(1)
    import importlib
    stage = sys.argv.pop(1)
    mod, fn = STAGES[stage]
    getattr(importlib.import_module(mod), fn)()


if __name__ == "__main__":
    main()
