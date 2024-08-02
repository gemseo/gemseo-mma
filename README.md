<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# gemseo-mma

[![PyPI - License](https://img.shields.io/pypi/l/gemseo)](https://www.gnu.org/licenses/lgpl-3.0.en.html)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/gemseo-mma)](https://pypi.org/project/gemseo-mma/)
[![PyPI](https://img.shields.io/pypi/v/gemseo-mma)](https://pypi.org/project/gemseo-mma/)
[![Codecov branch](https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo-mma/develop)](https://app.codecov.io/gl/gemseo:dev/gemseo-mma)

## Overview

A gemseo wrapper of Python version of Method of Moving Asymptothes in the implementation of Arjen Deetman.

## Installation

Install the latest stable version with `pip install gemseo-mma`.

Install the development version with
`pip install gemseo-mma@git+https://gitlab.com/gemseo/dev/gemseo-mma.git@develop`.

See [pip](https://pip.pypa.io/en/stable/getting-started/) for more information.

## Bugs and questions

Please use the [gitlab issue tracker](https://gitlab.com/gemseo/dev/gemseo-mma/-/issues)
to submit bugs or questions.

## Contributing

See the [contributing section of GEMSEO](https://gemseo.readthedocs.io/en/stable/software/developing.html#dev).

## Contributors

- Simone Coniglio
- Antoine Dechaume
- Original implementation of Arjen Deetman, see
    <https://github.com/arjendeetman/GCMMA-MMA-Python>.

## References

Svanberg, K. (1987). The Method of Moving Asymptotes -- A new method for
structural optimization. International Journal for Numerical Methods in
Engineering 24, 359-373. <doi:10.1002/nme.1620240207>, see
<https://onlinelibrary.wiley.com/doi/abs/10.1002/nme.1620240207>.

Svanberg, K. (n.d.). MMA and GCMMA -- two methods for nonlinear
optimization. Retrieved August 3, 2017 from
<https://people.kth.se/~krille/mmagcmma.pdf>
