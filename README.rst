..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

A GEMSEO wrapper of the Method of Moving Asymptotes implementation of Arjen Deetman.

Documentation
-------------

See https://gemseo.readthedocs.io/en/stable/plugins.html.

Bugs/Questions
--------------

Please use the gitlab issue tracker at
https://gitlab.com/gemseo/dev/gemseo-mma/-/issues
to submit bugs or questions.

License
-------

The GEMSEO-MMA source code is distributed under the GNU LGPL v3.0 license.
A copy of it can be found in the LICENSE.txt file.
The GNU LGPL v3.0 license is an exception to the GNU GPL v3.0 license.
A copy of the GNU GPL v3.0 license can be found in the LICENSES folder.

The GEMSEO-MMA examples are distributed under the BSD 0-Clause, a permissive
license that allows to copy paste the code of examples without preserving the
copyright mentions.

The GEMSEO-MMA documentation is distributed under the CC BY-SA 4.0 license.

The GEMSEO-MMA product depends on other software which have various licenses.
The list of dependencies with their licenses is given in the CREDITS.rst file.

Installation
------------

pip install gemseo-mma

Usage
-----

Like any other gemseo wrapped solver, MMA solver can be called setting the algo option to ``"MMA"``.
This algorithm can be used for single objective continuous optimization problem with non-linear inequality constraints.

Advanced options:

* tol: The KKT residual norm tolerance. This is not the one implemented in GEMSEO as it uses the local functions to be computed.

* max_optimization_step: Also known as ``move`` parameter control the maximum distance of the next iteration design point from the current one. Reducing this parameter avoid divergence for highly non-linear problems.

* min_asymptote_distance: The minimum distance of the asymptotes from the current design variable value.

* max_asymptote_distance: The maximum distance of the asymptotes from the current design variable value.

* asyinit: The initial asymptote distance from the current design variable value.

* asyincr The incremental factor of asymptote distance from the current design variable value for successful iterations.

* asydecr: The decremental factor of asymptote distance from the current design variable value for successful iterations.

* conv_tol: If provided control all other convergence tolerances.

The shortest is the distance of the asymptotes, the highest is the convexity of the local approximation.
It's another mechanism to control the optimization step.
You can find an example in examples/analytic_example.ipynb.

Contributors and acknowledgment
-------------------------------

* Simone Coniglio
* Antoine Dechaume
* Original implementation of Arjen Deetman, see https://github.com/arjendeetman/GCMMA-MMA-Python.

References
----------

Svanberg, K. (1987). The Method of Moving Asymptotes – A new method for structural optimization. International Journal
for Numerical Methods in Engineering 24, 359-373. doi:10.1002/nme.1620240207,
see https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207.

Svanberg, K. (n.d.). MMA and GCMMA – two methods for nonlinear optimization. Retrieved August 3, 2017 from
https://people.kth.se/~krille/mmagcmma.pdf
