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

from gemseo_mma.opt.core.mma_optimizer import MMAOptimizer
from gemseo_mma.opt.mma_settings import MMASvanberg_Settings

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem


class MMASvanberg(BaseOptimizationLibrary[MMASvanberg_Settings]):
    """Svanberg Method of Moving Asymptotes optimization library."""

    ALGORITHM_INFOS: ClassVar[dict[str, Any]] = {
        "MMA": OptimizationAlgorithmDescription(
            algorithm_name="MMA",
            internal_algorithm_name="MMA",
            library_name="MMA",
            description="The Method of Moving Asymptotes",
            Settings=MMASvanberg_Settings,
            require_gradient=True,
            handle_inequality_constraints=True,
        )
    }

    def _run(
        self,
        problem: OptimizationProblem,
    ) -> tuple[Any, Any]:
        return MMAOptimizer(problem).optimize(**self._settings.model_dump())

    def _get_result(
        self,
        problem: OptimizationProblem,
        message: Any,
        status: Any,
        *args: Any,
    ) -> OptimizationResult:
        problem = self._problem
        database = problem.database
        if len(database) == 0:
            return OptimizationResult(
                optimizer_name=self.algo_name,
                message=message,
                status=status,
                n_obj_call=0,
            )

        x_opt = database.get_x_vect(-1)
        is_feas, _ = problem.history.check_design_point_is_feasible(x_opt)
        f_opt = database.get_function_value(
            function_name=problem.objective.name,
            x_vect_or_iteration=x_opt,
        )
        c_opt = {
            cont.name: database.get_function_value(
                function_name=cont.name,
                x_vect_or_iteration=x_opt,
            )
            for cont in problem.constraints
        }
        c_opt_grad = {
            constr.name: database.get_gradient_history(function_name=constr.name)[-1]
            for constr in problem.constraints
        }
        if f_opt is not None and not problem.minimize_objective:
            f_opt = -f_opt

        return OptimizationResult(
            x_0=database.get_x_vect(1),
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
