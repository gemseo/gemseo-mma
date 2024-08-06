<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
# User guide

Like any other gemseo wrapped solver, MMA solver can be called setting
the algo option to `"MMA"`. This algorithm can be used for single
objective continuous optimization problem with non-linear inequality
constraints.

Advanced options:

- `tol`: The KKT residual norm tolerance. This is not the one
    implemented in GEMSEO as it uses the local functions to be computed.
- `max_optimization_step`: Also known as `move` parameter control the
    maximum distance of the next iteration design point from the current
    one. Reducing this parameter avoid divergence for highly non-linear
    problems.
- `min_asymptote_distance`: The minimum distance of the asymptotes from
    the current design variable value.
- `max_asymptote_distance`: The maximum distance of the asymptotes from
    the current design variable value.
- `initial_asymptotes_distance`: The initial asymptote distance from the
    current design variable value.
- `asymptotes_distance_amplification_coefficient` The incremental factor
    of asymptote distance from the current design variable value for
    successful iterations.
- `asymptotes_distance_reduction_coefficient`: The decremental factor of
    asymptote distance from the current design variable value for
    successful iterations.
- `conv_tol`: If provided control all other convergence tolerances.

The shortest is the distance of the asymptotes, the highest is the
convexity of the local approximation. It's another mechanism to control
the optimization step.
You can find this [example](https://gitlab.com/gemseo/dev/gemseo-mma/-/blob/develop/examples/analytic_example.ipynb).
