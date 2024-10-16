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
"""MMA optimizer library."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.algos.opt.base_optimization_library import BaseOptimizationLibrary
from gemseo.algos.opt.base_optimization_library import OptimizationAlgorithmDescription
from gemseo.algos.optimization_result import OptimizationResult

from gemseo_mma.opt._mma_settings import MMASvanbergSettings
from gemseo_mma.opt.core.mma_optimizer import MMAOptimizer

if TYPE_CHECKING:
    from gemseo.algos.base_problem import BaseProblem


class MMASvanberg(BaseOptimizationLibrary):
    """Svanberg Method of Moving Asymptotes optimization library."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "MMA": OptimizationAlgorithmDescription(
            "MMA",
            "MMA",
            "MMA",
            require_gradient=True,
            handle_equality_constraints=False,
            handle_inequality_constraints=True,
            positive_constraints=False,
            Settings=MMASvanbergSettings,
        )
    }

    def _run(self, problem: BaseProblem, **options: float | str) -> OptimizationResult:
        """Runs the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options dict for the algorithm,
                see associated MMA_options.json file.

        Returns:
            The OptimizationResult object.
        """
        optimizer = MMAOptimizer(self.problem)
        message, status = optimizer.optimize(**options)
        return self.get_optimum_from_database(message, status)

    def get_optimum_from_database(
        self, message: str | None = None, status: int | None = None
    ) -> OptimizationResult:
        """Get optimum from database using last point of database.

        Retrieves the optimum from the database and builds an optimization result object
        from it.

        Args:
            message: The solver message.
            status: The solver status.

        Returns:
            The OptimizationResult object.
        """
        problem = self.problem
        if len(problem.database) == 0:
            return OptimizationResult(
                optimizer_name=self.algo_name,
                message=message,
                status=status,
                n_obj_call=0,
            )
        x_0 = problem.database.get_x_vect(1)
        # get last point as optimum
        x_opt = problem.database.get_x_vect(-1)
        is_feas, _ = problem.history.check_design_point_is_feasible(x_opt)
        f_opt = problem.database.get_function_value(
            function_name=problem.objective.name, x_vect_or_iteration=x_opt
        )
        c_opt = {
            cont.name: problem.database.get_function_value(
                function_name=cont.name, x_vect_or_iteration=x_opt
            )
            for cont in problem.constraints
        }
        c_opt_grad = {
            cont.name: problem.database.get_gradient_history(function_name=cont.name)[
                -1
            ]
            for cont in problem.constraints
        }
        if f_opt is not None and not problem.minimize_objective:
            f_opt = -f_opt
        return OptimizationResult(
            x_0=x_0,
            x_opt=x_opt,
            f_opt=f_opt,
            optimizer_name=self.algo_name,
            message=message,
            status=status,
            n_obj_call=problem.objective.n_calls,
            is_feasible=is_feas,
            constraint_values=c_opt,
            constraints_grad=c_opt_grad,
        )
