# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Settings for the MMASvanberg algorithm."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from gemseo.algos.opt._base_optimization_library_settings import (
    BaseOptimizationLibrarySettings,
)
from gemseo.algos.opt._gradient_based_algorithm_settings import (
    GradientBasedAlgorithmSettings,
)
from gemseo.utils.pydantic import copy_field
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TCH002
from pydantic import model_validator

if TYPE_CHECKING:
    from typing_extensions import Self


copy_field_opt = partial(copy_field, model=BaseOptimizationLibrarySettings)


class MMASvanbergSettings(
    BaseOptimizationLibrarySettings, GradientBasedAlgorithmSettings
):
    """The settings for the MMA Svanberg algorithm."""

    ftol_abs: NonNegativeFloat = copy_field_opt("ftol_abs", default=1e-14)

    xtol_abs: NonNegativeFloat = copy_field_opt("xtol_abs", default=1e-14)

    ftol_rel: NonNegativeFloat = copy_field_opt("ftol_rel", default=1e-8)

    xtol_rel: NonNegativeFloat = copy_field_opt("xtol_rel", default=1e-8)

    ineq_tolerance: NonNegativeFloat = copy_field_opt("ineq_tolerance", default=1e-2)

    tol: NonNegativeFloat = Field(
        description=(
            """The tolerance of convergence used in MMA to be compared with the KKT
            residual."""
        ),
        default=1e-2,
    )

    conv_tol: NonNegativeFloat | None = Field(
        description=(
            """If provided, control all other convergence tolerances.

            Otherwise, the other convergence tolerances are used.
            """
        ),
        default=None,
    )

    max_optimization_step: NonNegativeFloat = Field(
        description="The maximum optimization step.", default=0.1
    )

    max_asymptote_distance: NonNegativeFloat = Field(
        description=(
            """The maximum distance of the asymptotes from the current design variable
            value."""
        ),
        default=10.0,
    )

    min_asymptote_distance: NonNegativeFloat = Field(
        description=(
            """The minimum distance of the asymptotes from the current design variable
            value."""
        ),
        default=0.01,
    )

    initial_asymptotes_distance: NonNegativeFloat = Field(
        description=(
            """The initial asymptotes distance from the current design variable
            value."""
        ),
        default=0.5,
    )

    asymptotes_distance_amplification_coefficient: NonNegativeFloat = Field(
        description="The amplification factor for successful iterations.",
        default=1.2,
    )

    asymptotes_distance_reduction_coefficient: NonNegativeFloat = Field(
        description="The decremental factor for unsuccessful iterations.", default=0.7
    )

    @model_validator(mode="after")
    def __check_tolerances(self) -> Self:
        """Check if a conv_tol was provided.

        Modify the other convergence values if needed.
        """
        if self.conv_tol is not None:
            self.ftol_rel = self.conv_tol
            self.ftol_abs = self.conv_tol
            self.xtol_rel = self.conv_tol
            self.xtol_abs = self.conv_tol
        else:
            self.conv_tol = min(
                self.ftol_rel, self.ftol_abs, self.xtol_rel, self.xtol_abs
            )
        return self
