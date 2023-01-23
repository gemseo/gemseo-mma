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
from __future__ import annotations

import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_result import OptimizationResult
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo_mma.opt.core.mma_optimizer import MMAOptimizer
from gemseo_mma.opt.lib_mma import MMASvanberg
from numpy import array
from numpy import ones


@pytest.fixture(params=[0.0, 0.25, 0.1, 1.0])
def x0(request):
    """Define an initial x guess fixture."""
    return request.param


@pytest.fixture(params=[0.0, 0.25, 0.1, 1.0])
def y0(request):
    """Define an initial y fixture."""
    return request.param


@pytest.fixture(params=[True, False])
def inactive_constraint(request):
    """Whether a supplementary constraint is provided (even if inactive)."""
    return request.param


@pytest.fixture(params=[True, False])
def maximization(request):
    """Whether a supplementary constraint is provided (even if inactive)."""
    return request.param


def obj_func(x=0.0, y=0.0):
    """The objective function."""
    f = (x - 1.0) ** 2 + (y - 1.0) ** 2
    return f


def d_obj_func(x=0.0, y=0.0):
    """The objective function jacobian."""
    jac = array([2.0 * (x[0] - 1.0), 2 * (y[0] - 1.0)])
    return jac


def obj_func_max(x=0.0, y=0.0):
    """The objective function for maximization."""
    f = -((x - 1.0) ** 2) - (y - 1.0) ** 2
    return f


def d_obj_func_max(x=0.0, y=0.0):
    """The objective function for maximization jacobian."""
    jac = array([-2.0 * (x[0] - 1.0), -2 * (y[0] - 1.0)])
    return jac


def cstr_func(x=0.0, y=0.0):
    """The inequality constraint function."""
    g = x + y - 1.0
    return g


def d_cstr_func(x=0.0, y=0.0):
    """The inequality constraint function jacobian."""
    jac = ones((1, 2))
    return jac


def cstr_func2(x=0.0, y=0.0):
    """The equality constraint function."""
    h = -(x**2) - y**2
    return h


def d_cstr_func2(x=0.0, y=0.0):
    """The equality constraint function jacobian."""
    jac = array([-2 * x[0], -2 * y[0]])
    return jac


@pytest.fixture
def analytical_test_2d_ineq(x0, y0, inactive_constraint, maximization):
    """Test for lagrange multiplier."""
    if maximization:
        disc1 = create_discipline(
            "AutoPyDiscipline", py_func=obj_func_max, py_jac=d_obj_func_max
        )
    else:
        disc1 = create_discipline(
            "AutoPyDiscipline", py_func=obj_func, py_jac=d_obj_func
        )

    disc2 = create_discipline("AutoPyDiscipline", py_func=cstr_func, py_jac=d_cstr_func)
    disc3 = create_discipline(
        "AutoPyDiscipline", py_func=cstr_func2, py_jac=d_cstr_func2
    )
    ds = DesignSpace()
    ds.add_variable("x", l_b=0.0, u_b=1.0, value=x0)
    ds.add_variable("y", l_b=0.0, u_b=1.0, value=y0)
    scenario = create_scenario(
        disciplines=[disc1, disc2, disc3],
        formulation="DisciplinaryOpt",
        objective_name="f",
        design_space=ds,
        maximize_objective=maximization,
    )
    if inactive_constraint:
        scenario.add_constraint("h", "ineq")
    scenario.add_constraint("g", "ineq")
    return scenario


parametrized_options = pytest.mark.parametrize(
    "options",
    [
        {
            "max_iter": 50,
            "algo_options": {"tol": 1e-4, "normalize_design_space": True},
        },
        {
            "max_iter": 30,
            "algo_options": {"conv_tol": 1e-16, "tol": 1e-16},
        },
        {
            "max_iter": 50,
            "algo_options": {
                "xtol_rel": 1e-7,
                "xtol_abs": 1e-7,
                "ftol_rel": 1e-7,
                "ftol_abs": 1e-7,
                "ctol_abs": 1e-7,
                "tol": 1e-16,
            },
        },
    ],
)
parametrized_algo_ineq = pytest.mark.parametrize("algo_ineq", ["MMA"])


@parametrized_options
@parametrized_algo_ineq
def test_execution_with_scenario(analytical_test_2d_ineq, options, algo_ineq):
    """Test for optimization scenario execution using MMA solver."""
    opt = options.copy()
    opt["algo"] = algo_ineq
    analytical_test_2d_ineq.execute(opt)
    problem = analytical_test_2d_ineq.formulation.opt_problem
    assert pytest.approx(problem.solution.x_opt, 1e-4) == array([0.5, 0.5])


def test_direct_execution(analytical_test_2d_ineq):
    """Test for optimization problem execution using MMA solver."""
    problem = analytical_test_2d_ineq.formulation.opt_problem
    optimizer = MMAOptimizer(problem)
    optimizer.optimize()
    assert pytest.approx(problem.design_space.get_current_value(), 1e-4) == array(
        [0.5, 0.5]
    )


def test_get_optimum_from_database(analytical_test_2d_ineq):
    """Test for get_optimum_from_database call before opt problem resolution."""
    lib = MMASvanberg()
    lib.problem = analytical_test_2d_ineq.formulation.opt_problem
    assert isinstance(lib.get_optimum_from_database(), OptimizationResult)
