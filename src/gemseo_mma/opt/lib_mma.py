# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""MMA optimizer library."""
from __future__ import annotations

from typing import Any

from gemseo.algos.opt.opt_lib import OptimizationAlgorithmDescription
from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult

from gemseo_mma.opt.core.mma_optimizer import MMAOptimizer


class MMASvanberg(OptimizationLibrary):
    """Svanberg Method of Moving Asymptotes optimization library."""

    descriptions: dict[str, OptimizationAlgorithmDescription]
    """The optimization algorithm description."""

    def __init__(self) -> None:
        """Constructor."""
        super().__init__()
        self.descriptions = {
            "MMA": OptimizationAlgorithmDescription(
                "MMA",
                "MMA",
                "MMA",
                require_gradient=True,
                handle_equality_constraints=False,
                handle_inequality_constraints=True,
                positive_constraints=False,
            )
        }

    def _get_options(
        self,
        max_iter: int = 1000,
        ftol_abs: float = 1e-14,
        xtol_abs: float = 1e-14,
        max_time: float = 0.0,
        ftol_rel: float = 1e-8,
        xtol_rel: float = 1e-8,
        ctol_abs: float | None = None,
        stopval: float | None = None,
        eq_tolerance: float = 1e-2,
        ineq_tolerance: float = 1e-4,
        tol: float = 1e-2,
        conv_tol: float = None,
        max_optimization_step: float = 0.1,
        max_asymptote_distance: float = 10.0,
        min_asymptote_distance: float = 0.01,
        asyinit: float = 0.5,
        asyincr: float = 1.2,
        asydecr: float = 0.7,
        normalize_design_space: bool = False,
    ) -> dict[str, Any]:
        r"""Sets the options.

        Args:
            ftol_abs: The absolute tolerance on the objective function.
            xtol_abs: The absolute tolerance on the design parameters.
            max_time: The maximum runtime in seconds. The value 0 means no runtime
                limit.
            max_iter: The maximum number of iterations.
            ftol_rel: The relative tolerance on the objective function.
            xtol_rel: The relative tolerance on the design parameters.
            ctol_abs: The absolute tolerance on the constraints.
            stopval: The objective value at which the optimization will stop.
                Stop minimizing when an objective value :math:`\leq` stopval is
                found, or stop maximizing when a value :math:`\geq` stopval
                is found. If None, this termination condition will not be active.
            normalize_design_space: If True, normalize the design variables between 0
                and 1.
            eq_tolerance: The tolerance on the equality constraints.
            ineq_tolerance: The tolerance on the inequality constraints.
            tol: tolerance of convergence used in MMA to be compared with kkt residual.
            conv_tol: If provided control all other convergence tolerances.
            max_optimization_step: The maximum optimization step.
            max_asymptote_distance: The maximum distance of the asymptotes from the
                current design variable value.
            min_asymptote_distance: The minimum distance of the asymptotes from the
                current design variable value.
            asyinit: The initial asymptotes distance from the current design variable
                value.
            asyincr: The incremental factor for successful iterations.
            asydecr: The decremental factor for unsuccessful iterations.

        Returns:
            The converted options.

        Raises:
            ValueError: If an option is invalid.
        """
        if conv_tol is not None:
            ftol_rel = conv_tol
            ftol_abs = conv_tol
            xtol_rel = conv_tol
            xtol_abs = conv_tol
        else:
            conv_tol = min(ftol_rel, ftol_abs, xtol_rel, xtol_abs)

        if ctol_abs is None:
            ctol_abs = conv_tol

        return self._process_options(
            max_iter=max_iter,
            tol=tol,
            normalize_design_space=normalize_design_space,
            conv_tol=conv_tol,
            max_optimization_step=max_optimization_step,
            max_asymptote_distance=max_asymptote_distance,
            min_asymptote_distance=min_asymptote_distance,
            asyinit=asyinit,
            asyincr=asyincr,
            asydecr=asydecr,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_time=max_time,
            stopval=stopval,
            eq_tolerance=eq_tolerance,
            ineq_tolerance=ineq_tolerance,
            ctol_abs=ctol_abs,
        )

    def _run(self, **options: float | int | str) -> OptimizationResult:
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
        x_0 = problem.database.get_x_by_iter(0)
        # get last point as optimum
        x_opt = problem.database.get_x_by_iter(-1)
        is_feas, _violation = problem.get_violation_criteria(x_opt)
        f_opt = problem.database.get_f_of_x(fname=problem.objective.name, x_vect=x_opt)
        c_opt = {
            cont.name: problem.database.get_f_of_x(fname=cont.name, x_vect=x_opt)
            for cont in problem.constraints
        }
        c_opt_grad = {
            cont.name: problem.database.get_func_grad_history(funcname=cont.name)[-1]
            for cont in problem.constraints
        }
        # f_opt, x_opt, is_feas, c_opt, c_opt_grad = problem.get_optimum()
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
            constraints_values=c_opt,
            constraints_grad=c_opt_grad,
        )
