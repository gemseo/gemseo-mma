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
"""MMA optimization solver."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy import atleast_2d
from numpy import ndarray

from gemseo_mma.opt.core.mma import compute_kkt_residual_on_local_approximation
from gemseo_mma.opt.core.mma import solve_mma_local_approximation_problem

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem

# Import MMA functions

LOGGER = logging.getLogger(__name__)


class MMAOptimizer:
    """Method of Moving Asymptotes optimizer class.

    This class run an optimization algorithm to solve Non-linear Optimization problems
    with constraints. The objective function and the constraints and their gradients are
    needed for the optimization algorithm. The original implementation the next
    iteration candidate is computed using mmasub function adapted from `
    https://github.com/arjendeetman/GCMMA-MMA-Python
    <https://github.com/arjendeetman/GCMMA-MMA-Python>`_.
    The external and internal move
    limit can be tuned to control minimum and maximum local approximation convexity. The
    max_optimization_step parameter can be used to control the optimization step. To
    avoid solver divergence in the case of highly non-linear problems one should use
    smaller values of the max_optimization_step, max_asymptote_distance and
    min_asymptote_distance.
    """

    __DEFAULT_NORMALIZE_DESIGN_SPACE = False
    __DEFAULT_MAX_OPTIM_STEP = 0.1
    __DEFAULT_MIN_DISTANCE = 0.01
    __DEFAULT_MAX_DISTANCE = 10.0
    __DEFAULT_TOLERANCE = 1e-6
    __DEFAULT_MAXITER = 1000
    __DEFAULT_KKT_TOL = 1e-2
    __EMPTY_STRING = ""
    __DEFAULT_ASYINIT = 0.5
    __DEFAULT_ASYINCR = 1.2
    __DEFAULT_ASYDECR = 0.7

    __max_optimization_step: float
    """The maximum optimization step."""
    __min_asymptote_distance: float
    """The minimum distance of the asymptotes from the current design variable value."""
    __max_asymptote_distance: float
    """The maximum distance of the asymptotes from the current design variable value."""
    __ftol_abs: float
    """The absolute tolerance on the objective function."""
    __ftol_rel: float
    """The relative tolerance on the objective function."""
    __xtol_rel: float
    """The relative tolerance on the design parameters."""
    __xtol_abs: float
    """The absolute tolerance on the design parameters."""
    __tol: float
    """KKT residual norm tolerance."""
    __max_iter: int
    """The maximum number of iterations."""
    __normalize_design_space: bool
    """Whether to normalize the design space."""
    __message: str
    """The message of the optimization problem."""
    __problem: OptimizationProblem
    """The GEMSEO OptimizationProblem to be solved."""
    __initial_asymptotes_distance: float
    """The initial asymptotes distance from the current design variable value."""
    __asymptotes_distance_amplification_coefficient: float
    """The amplification factor for successful iterations."""
    __asymptotes_distance_reduction_coefficient: float
    """The decremental factor for unsuccessful iterations."""
    __ineq_tolerance: float
    """The inequality constraints tolerance."""

    def __init__(self, problem: OptimizationProblem) -> None:
        """Constructor."""
        self.__problem = problem
        self.__message = self.__EMPTY_STRING
        self.__normalize_design_space = self.__DEFAULT_NORMALIZE_DESIGN_SPACE
        self.__max_iter = self.__DEFAULT_MAXITER
        self.__tol = self.__DEFAULT_KKT_TOL
        self.__xtol_abs = self.__DEFAULT_TOLERANCE
        self.__xtol_rel = self.__DEFAULT_TOLERANCE
        self.__ftol_rel = self.__DEFAULT_TOLERANCE
        self.__ftol_abs = self.__DEFAULT_TOLERANCE
        self.__ineq_tolerance = self.__DEFAULT_TOLERANCE
        self.__max_asymptote_distance = self.__DEFAULT_MAX_DISTANCE
        self.__min_asymptote_distance = self.__DEFAULT_MIN_DISTANCE
        self.__max_optimization_step = self.__DEFAULT_MAX_OPTIM_STEP
        self.__initial_asymptotes_distance = self.__DEFAULT_ASYINIT
        self.__asymptotes_distance_amplification_coefficient = self.__DEFAULT_ASYINCR
        self.__asymptotes_distance_reduction_coefficient = self.__DEFAULT_ASYDECR

    def optimize(self, **options: bool | int | float) -> tuple[str, int]:
        """Optimize the problem.

        Args:
            **options: The optimization problem options.

        Returns:
            The optimization solver message and final status.
        """
        self.__normalize_design_space = options.get(
            "normalize_design_space", self.__DEFAULT_NORMALIZE_DESIGN_SPACE
        )
        self.__max_iter = options.get("max_iter", self.__DEFAULT_MAXITER)
        self.__tol = options.get("tol", self.__DEFAULT_TOLERANCE)
        self.__xtol_abs = options.get("xtol_abs", self.__DEFAULT_TOLERANCE)
        self.__xtol_rel = options.get("xtol_rel", self.__DEFAULT_TOLERANCE)
        self.__ftol_rel = options.get("ftol_rel", self.__DEFAULT_TOLERANCE)
        self.__ftol_abs = options.get("ftol_abs", self.__DEFAULT_TOLERANCE)
        self.__max_asymptote_distance = options.get(
            "max_asymptote_distance", self.__DEFAULT_MAX_DISTANCE
        )
        self.__min_asymptote_distance = options.get(
            "min_asymptote_distance", self.__DEFAULT_MIN_DISTANCE
        )
        self.__max_optimization_step = options.get(
            "max_optimization_step", self.__DEFAULT_MAX_OPTIM_STEP
        )
        self.__initial_asymptotes_distance = options.get(
            "initial_asymptotes_distance", self.__DEFAULT_ASYINIT
        )
        self.__asymptotes_distance_amplification_coefficient = options.get(
            "asymptotes_distance_amplification_coefficient", self.__DEFAULT_ASYINCR
        )
        self.__asymptotes_distance_reduction_coefficient = options.get(
            "asymptotes_distance_reduction_coefficient", self.__DEFAULT_ASYDECR
        )
        self.__ineq_tolerance = options.get("ineq_tolerance", self.__DEFAULT_TOLERANCE)

        # initialize database
        if not self.__normalize_design_space:
            x0 = self.__problem.design_space.get_current_value()
        else:
            x0 = self.__problem.design_space.normalize_vect(
                self.__problem.design_space.get_current_value()
            )

        # launch optim
        xopt = self.iterate(x0)[0]
        self.__problem.design_space.set_current_value(xopt)

        return self.__message, 0

    def iterate(self, x0: ndarray) -> tuple[ndarray, ndarray]:
        """Iterate until convergence from the starting guess.

        Args:
            x0: The starting guess design point.

        Returns:
            The optimum design point and objective value.
        """
        n = len(x0)
        eeen = np.ones((n, 1))
        xval = np.reshape(x0, eeen.shape)
        xold1 = xval.copy()
        xold2 = xval.copy()
        if not self.__normalize_design_space:
            xmin = np.reshape(
                self.__problem.design_space.get_lower_bounds(), eeen.shape
            )
            xmax = np.reshape(
                self.__problem.design_space.get_upper_bounds(), eeen.shape
            )
        else:
            xmin = 0 * eeen
            xmax = eeen
        low = xmin.copy()
        upp = xmax.copy()

        a0 = 1

        outeriter = 0
        maxoutit = self.__max_iter
        kkttol = self.__tol
        (
            f0val,
            df0dx,
            fval,
            dfdx,
        ) = self.__get_objective_and_constraints_with_gradients(xval.flatten())
        m = len(fval)
        eeem = np.ones((m, 1))
        zerom = np.zeros((m, 1))
        c = 1000 * eeem
        d = eeem.copy()
        a = zerom.copy()
        # The iterations starts
        kktnorm = kkttol + 10
        f_ref = f0val
        x_ref = np.linalg.norm(xval)
        change_fc = 10
        change_f = 10
        change_x = 10
        change_relative_f = 10
        change_relative_x = 10
        outit = 0
        while (
            (kktnorm > kkttol)
            and (change_x > self.__xtol_abs)
            and (change_relative_x > self.__xtol_rel)
            and (change_relative_f > self.__ftol_rel)
            and (change_f > self.__ftol_abs)
            or (any(fval > self.__ineq_tolerance) and change_fc > self.__ftol_abs)
        ) and (outit < maxoutit):
            outit += 1
            outeriter += 1

            (
                xmma,
                ymma,
                zmma,
                lam,
                xsi,
                eta,
                mu,
                zet,
                s,
                low,
                upp,
            ) = solve_mma_local_approximation_problem(
                m,
                n,
                outeriter,
                xval,
                xmin,
                xmax,
                xold1,
                xold2,
                f0val,
                df0dx,
                fval,
                dfdx,
                low,
                upp,
                a0,
                a,
                c,
                d,
                self.__max_optimization_step,
                self.__max_asymptote_distance,
                self.__min_asymptote_distance,
                self.__initial_asymptotes_distance,
                self.__asymptotes_distance_amplification_coefficient,
                self.__asymptotes_distance_reduction_coefficient,
            )
            (
                f0valnew,
                df0dxnew,
                fvalnew,
                dfdxnew,
            ) = self.__get_objective_and_constraints_with_gradients(xmma.flatten())
            change_x = np.linalg.norm(xval - xmma)
            change_f = np.abs(f0valnew - f0val)
            change_fc = np.abs(fvalnew.max() - fval.max())
            change_relative_x = change_x / x_ref
            change_relative_f = change_f / f_ref
            # Some vectors are updated:
            xold2 = xold1.copy()
            xold1 = xval.copy()
            xval = xmma.copy()
            # Re-calculate function values and gradients of the objective and
            # constraints functions
            f0val, df0dx, fval, dfdx = f0valnew, df0dxnew, fvalnew, dfdxnew
            # The residual vector of the KKT conditions is calculated
            _, kktnorm, _ = compute_kkt_residual_on_local_approximation(
                m,
                n,
                xmma,
                ymma,
                zmma,
                lam,
                xsi,
                eta,
                mu,
                zet,
                s,
                xmin,
                xmax,
                df0dx,
                fval,
                dfdx,
                a0,
                a,
                c,
                d,
            )

        if self.__normalize_design_space:
            xopt = self.__problem.design_space.unnormalize_vect(xval.flatten())
        else:
            xopt = xval.flatten()

        LOGGER.info("END OF MMA ALGORITHM")
        if kktnorm <= kkttol:
            self.__message = "KKT norm criteria met"
        else:
            self.__message = "change criteria met"
        return xopt, f0val

    def __get_objective_and_constraints_with_gradients(
        self, xval: ndarray
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Compute the objective and constraints with their gradients.

        Args:
            xval: The current design value.

        Returns:
            The objective and constraint value in the provided design point and their
                gradients.
        """
        f0val = self.__problem.objective(xval)
        df0dx = self.__problem.objective.jac(xval)
        df0dx = np.reshape(df0dx, (df0dx.size, 1))
        constraint_list = []
        constraint_jac_list = []
        for constraint in self.__problem.constraints:
            constraint_list.append(constraint(xval).flatten())
            constraint_jac_list.append(atleast_2d(constraint.jac(xval)))
        fval = np.concatenate(constraint_list)
        fval = np.reshape(fval, (fval.size, 1))
        dfdx = np.concatenate(constraint_jac_list, axis=0)
        return f0val, df0dx, fval, dfdx
