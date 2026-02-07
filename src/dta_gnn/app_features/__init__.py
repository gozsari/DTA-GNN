"""App/UI feature helpers.

Historically, some entrypoints imported plotting helpers from
`dta_gnn.app_features`. The codebase now keeps visualization helpers in
`dta_gnn.visualization`, while UI-specific utilities live under
`dta_gnn.app_features.*`.

This package preserves the legacy public surface while enabling submodules like
`dta_gnn.app_features.compound` and `dta_gnn.app_features.proteins`.
"""

from __future__ import annotations

from dta_gnn.visualization import (
    plot_activity_distribution,
    plot_chemical_space,
    plot_split_sizes,
)

__all__ = [
    "plot_activity_distribution",
    "plot_chemical_space",
    "plot_split_sizes",
]
