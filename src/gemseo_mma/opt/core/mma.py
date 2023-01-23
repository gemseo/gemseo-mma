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
"""GCMMA-MMA-Python. This file is part of GCMMA-MMA-Python.

GCMMA-MMA-Python is licensed
under the terms of GNU General Public License as published by the Free Software
Foundation. For more information and the LICENSE file, see `https://github.com/
arjendeetman/GCMMA-MMA-Python <https://github.com/arjendeetman/GCMMA-MMA-Python>`_
. The original work is written by
Krister Svanberg in MATLAB. This is the python version of the code written Arjen
Deetman. version 09-11-2019.

MMA optimizer.

Original work written by Krister Svanberg in Matlab. This is the python version of the
code written by Arjen Deetman.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray
from numpy.linalg import lstsq
from scipy.linalg import solve  # or use numpy: from numpy.linalg import solve
from scipy.sparse import diags  # or use numpy: from numpy import diag as diags


# Function for the MMA sub problem
def solve_mma_local_approximation_problem(
    m: int,
    n: int,
    iter: int,
    xval: ndarray,
    xmin: ndarray,
    xmax: ndarray,
    xold1: ndarray,
    xold2: ndarray,
    f0val: ndarray,
    df0dx: ndarray,
    fval: ndarray,
    dfdx: ndarray,
    low: ndarray,
    upp: ndarray,
    a0: float,
    a: ndarray,
    c: ndarray,
    d: ndarray,
    move: float,
    external_move_limit: float = 10.0,
    internal_limit: float = 0.01,
    asyinit: float = 0.5,
    asyincr: float = 1.2,
    asydecr: float = 0.7,
) -> tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
]:
    """MMA sub function.

    This function mmasub performs one MMA-iteration, aimed at solving the nonlinear
    programming problem:

    Minimize    f_0(x) + a_0*z + sum( c_i*y_i + 0.5*d_i*(y_i)^2 )
    subject to  f_i(x) - a_i*z - y_i <= 0,  i = 1,...,m
                xmin_j <= x_j <= xmax_j,    j = 1,...,n
                z >= 0,   y_i >= 0,         i = 1,...,m

    Args:
        m: The number of general constraints.
        n: The number of variables x_j.
        iter: The current iteration number ( =1 the first time mmasub is called).
        xval: The column vector with the current values of the variables x_j.
        xmin: The column vector with the lower bounds for the variables x_j.
        xmax: The column vector with the upper bounds for the variables x_j.
        xold1: The value of xval, one iteration ago (provided that iter>1).
        xold2: The value of xval, two iterations ago (provided that iter>2).
        f0val: The value of the objective function f_0 at xval.
        df0dx: The column vector with the derivatives of the objective function
                f_0 with respect to the variables x_j, calculated at xval.
        fval: The column vector with the values of the constraint functions f_i,
            calculated at xval.
        dfdx: The (m x n)-matrix with the derivatives of the constraint functions
                f_i with respect to the variables x_j, calculated at xval.
        low: The column vector with the lower asymptotes from the previous
                iteration (provided that iter>1).
        upp: The column vector with the upper asymptotes from the previous
                iteration (provided that iter>1).
        a0: The constants a_0 in the term a_0*z.
        a: The column vector with the constants a_i in the terms a_i*z.
        c: The column vector with the constants c_i in the terms c_i*y_i.
        d: The column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
        move: The maximum optimization step.
        external_move_limit: The maximum distance of the asymptotes from the current
            design variable value.
        internal_limit: The minimum distance of the asymptotes from the current design
            variable value.
        asyinit: The initial asymptotes distance from the current design variable value.
        asyincr: The incremental factor for successful iterations.
        asydecr: The decremental factor for unsuccessful iterations.

    Returns:
        xmma  = The Column vector with the optimal values of the variables x_j
                in the current MMA subproblem.
        ymma  = The Column vector with the optimal values of the variables y_i
                in the current MMA subproblem.
        zmma  = The Scalar with the optimal value of the variable z
                in the current MMA subproblem.
        lam   = The Lagrange multipliers for the m general MMA constraints.
        xsi   = The Lagrange multipliers for the n constraints alfa_j - x_j <= 0.
        eta   = The Lagrange multipliers for the n constraints x_j - beta_j <= 0.
        mu    = The Lagrange multipliers for the m constraints -y_i <= 0.
        zet   = The Lagrange multiplier for the single constraint -z <= 0.
        s     = The Slack variables for the m general MMA constraints.
        low   = The Column vector with the lower asymptotes, calculated and used
                in the current MMA subproblem.
        upp   = The Column vector with the upper asymptotes, calculated and used
                in the current MMA subproblem.
    """
    epsimin = 0.0000001
    raa0 = 0.00001
    albefa = 0.1
    eeen = np.ones((n, 1))
    eeem = np.ones((m, 1))
    zeron = np.zeros((n, 1))
    # Calculation of the asymptotes low and upp
    if iter <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = eeen.copy()
        factor[np.where(zzz > 0)] = asyincr
        factor[np.where(zzz < 0)] = asydecr
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        lowmin = xval - external_move_limit * (xmax - xmin)
        lowmax = xval - internal_limit * (xmax - xmin)
        uppmin = xval + internal_limit * (xmax - xmin)
        uppmax = xval + external_move_limit * (xmax - xmin)
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)
    # Calculation of the bounds alfa and beta
    zzz1 = low + albefa * (xval - low)
    zzz2 = xval - move * (xmax - xmin)
    zzz = np.maximum(zzz1, zzz2)
    alfa = np.maximum(zzz, xmin)
    zzz1 = upp - albefa * (upp - xval)
    zzz2 = xval + move * (xmax - xmin)
    zzz = np.minimum(zzz1, zzz2)
    beta = np.minimum(zzz, xmax)
    # Calculations of p0, q0, pp, qq and b
    xmami = xmax - xmin
    xmamieps = 0.00001 * eeen
    xmami = np.maximum(xmami, xmamieps)
    xmamiinv = eeen / xmami
    ux1 = upp - xval
    ux2 = ux1 * ux1
    xl1 = xval - low
    xl2 = xl1 * xl1
    uxinv = eeen / ux1
    xlinv = eeen / xl1
    p0 = zeron.copy()
    q0 = zeron.copy()
    p0 = np.maximum(df0dx, 0)
    q0 = np.maximum(-df0dx, 0)
    pq0 = 0.001 * (p0 + q0) + raa0 * xmamiinv
    p0 = p0 + pq0
    q0 = q0 + pq0
    p0 = p0 * ux2
    q0 = q0 * xl2
    pp = np.zeros((m, n))
    qq = np.zeros((m, n))
    pp = np.maximum(dfdx, 0)
    qq = np.maximum(-dfdx, 0)
    ppqq = 0.001 * (pp + qq) + raa0 * np.dot(eeem, xmamiinv.T)
    pp = pp + ppqq
    qq = qq + ppqq
    pp = (diags(ux2.flatten(), 0).dot(pp.T)).T
    qq = (diags(xl2.flatten(), 0).dot(qq.T)).T
    b = np.dot(pp, uxinv) + np.dot(qq, xlinv) - fval
    xmma, ymma, zmma, lam, xsi, eta, mu, zet, s = __subsolv(
        m, n, epsimin, low, upp, alfa, beta, p0, q0, pp, qq, a0, a, b, c, d
    )
    # Return values
    return xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp


# Function for solving the subproblem (can be used for MMA)
def __subsolv(
    m: int,
    n: int,
    epsimin: float,
    low: ndarray,
    upp: ndarray,
    alfa: ndarray,
    beta: ndarray,
    p0: ndarray,
    q0: ndarray,
    pp: ndarray,
    qq: ndarray,
    a0: ndarray,
    a: ndarray,
    b: ndarray,
    c: ndarray,
    d: ndarray,
) -> tuple[
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray,
    ndarray | ndarray,
]:
    """Solve the subproblem by a primal-dual Newton method.

    This function subsolv solves the MMA subproblem:

    minimize SUM[p0j/(uppj-xj) + q0j/(xj-lowj)] + a0*z + SUM[ci*yi + 0.5*di*(yi)^2],

    subject to SUM[pij/(uppj-xj) + qij/(xj-lowj)] - ai*z - yi <= bi,
        alfaj <=  xj <=  betaj,  yi >= 0,  z >= 0.

    Args:
        m: The number of general constraints.
        n: The number of variables x_j.
        epsimin:The convergence tolerance of the local approximation sub problem.
        low: The column vector with the lower asymptotes, calculated and used in the
            current MMA subproblem.
        upp: The column vector with the upper asymptotes, calculated and used in the
            current MMA subproblem.
        alfa: The column vector of local lower bound alfa_j.
        beta: The column vector of local upper bound beta_j.
        p0: The column vector of p0_j.
        q0: The column vector of q0_j.
        pp: The m x n matrix of p_ij.
        qq: The m x n matrix of q_ij.
        a0: The constants a_0 in the term a_0*z.
        a: The column vector with the constants a_i in the terms a_i*z.
        c: The column vector with the constants c_i in the terms c_i*y_i.
        d: The column vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.
        b: The column vector of b_i.

    Returns:
        All the unkowns of the subproblem.
    """
    een = np.ones((n, 1))
    eem = np.ones((m, 1))
    epsi = 1
    x = 0.5 * (alfa + beta)
    y = eem.copy()
    z = np.array([[1.0]])
    lam = eem.copy()
    xsi = een / (x - alfa)
    xsi = np.maximum(xsi, een)
    eta = een / (beta - x)
    eta = np.maximum(eta, een)
    mu = np.maximum(eem, 0.5 * c)
    zet = np.array([[1.0]])
    s = eem.copy()
    itera = 0
    # Start while epsi>epsimin
    while epsi > epsimin:
        epsvecn = epsi * een
        epsvecm = epsi * eem
        ux1 = upp - x
        xl1 = x - low
        ux2 = ux1 * ux1
        xl2 = xl1 * xl1
        uxinv1 = een / ux1
        xlinv1 = een / xl1
        plam = p0 + np.dot(pp.T, lam)
        qlam = q0 + np.dot(qq.T, lam)
        gvec = np.dot(pp, uxinv1) + np.dot(qq, xlinv1)
        dpsidx = plam / ux2 - qlam / xl2
        rex = dpsidx - xsi + eta
        rey = c + d * y - mu - lam
        rez = a0 - zet - np.dot(a.T, lam)
        relam = gvec - a * z - y + s - b
        rexsi = xsi * (x - alfa) - epsvecn
        reeta = eta * (beta - x) - epsvecn
        remu = mu * y - epsvecm
        rezet = zet * z - epsi
        res = lam * s - epsvecm
        residu1 = np.concatenate((rex, rey, rez), axis=0)
        residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
        residu = np.concatenate((residu1, residu2), axis=0)
        residunorm = np.sqrt((np.dot(residu.T, residu)).item())
        residumax = np.max(np.abs(residu))
        ittt = 0
        # Start while (residumax>0.9*epsi) and (ittt<200)
        while (residumax > 0.9 * epsi) and (ittt < 200):
            ittt = ittt + 1
            itera = itera + 1
            ux1 = upp - x
            xl1 = x - low
            ux2 = ux1 * ux1
            xl2 = xl1 * xl1
            ux3 = ux1 * ux2
            xl3 = xl1 * xl2
            uxinv1 = een / ux1
            xlinv1 = een / xl1
            uxinv2 = een / ux2
            xlinv2 = een / xl2
            plam = p0 + np.dot(pp.T, lam)
            qlam = q0 + np.dot(qq.T, lam)
            gvec = np.dot(pp, uxinv1) + np.dot(qq, xlinv1)
            gg = (diags(uxinv2.flatten(), 0).dot(pp.T)).T - (
                diags(xlinv2.flatten(), 0).dot(qq.T)
            ).T
            dpsidx = plam / ux2 - qlam / xl2
            delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
            dely = c + d * y - lam - epsvecm / y
            delz = a0 - np.dot(a.T, lam) - epsi / z
            dellam = gvec - a * z - y - b + epsvecm / lam
            diagx = plam / ux3 + qlam / xl3
            diagx = 2 * diagx + xsi / (x - alfa) + eta / (beta - x)
            diagxinv = een / diagx
            diagy = d + mu / y
            diagyinv = eem / diagy
            diaglam = s / lam
            diaglamyi = diaglam + diagyinv
            # Start if m<n
            if m < n:
                blam = dellam + dely / diagy - np.dot(gg, (delx / diagx))
                bb = np.concatenate((blam, delz), axis=0)
                alam = np.asarray(
                    diags(diaglamyi.flatten(), 0)
                    + (diags(diagxinv.flatten(), 0).dot(gg.T).T).dot(gg.T)
                )
                aar1 = np.concatenate((alam, a), axis=1)
                aar2 = np.concatenate((a, -zet / z), axis=0).T
                aa = np.concatenate((aar1, aar2), axis=0)
                solut = solve(aa, bb)
                dlam = solut[0:m]
                dz = solut[m : m + 1]
                dx = -delx / diagx - np.dot(gg.T, dlam) / diagx
            else:
                diaglamyiinv = eem / diaglamyi
                dellamyi = dellam + dely / diagy
                axx = np.asarray(
                    diags(diagx.flatten(), 0)
                    + (diags(diaglamyiinv.flatten(), 0).dot(gg).T).dot(gg)
                )
                azz = zet / z + np.dot(a.T, (a / diaglamyi))
                axz = np.dot(-gg.T, (a / diaglamyi))
                bx = delx + np.dot(gg.T, (dellamyi / diaglamyi))
                bz = delz - np.dot(a.T, (dellamyi / diaglamyi))
                aar1 = np.concatenate((axx, axz), axis=1)
                aar2 = np.concatenate((axz.T, azz), axis=1)
                aa = np.concatenate((aar1, aar2), axis=0)
                bb = np.concatenate((-bx, -bz), axis=0)
                solut, _i, _j, _k = lstsq(aa, bb, rcond=-1)
                dx = solut[0:n]
                dz = solut[n : n + 1]
                dlam = (
                    np.dot(gg, dx) / diaglamyi
                    - dz * (a / diaglamyi)
                    + dellamyi / diaglamyi
                )
                # End if m<n
            dy = -dely / diagy + dlam / diagy
            dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
            deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
            dmu = -mu + epsvecm / y - (mu * dy) / y
            dzet = -zet + epsi / z - zet * dz / z
            ds = -s + epsvecm / lam - (s * dlam) / lam
            xx = np.concatenate((y, z, lam, xsi, eta, mu, zet, s), axis=0)
            dxx = np.concatenate((dy, dz, dlam, dxsi, deta, dmu, dzet, ds), axis=0)
            #
            stepxx = -1.01 * dxx / xx
            stmxx = np.max(stepxx)
            stepalfa = -1.01 * dx / (x - alfa)
            stmalfa = np.max(stepalfa)
            stepbeta = 1.01 * dx / (beta - x)
            stmbeta = np.max(stepbeta)
            stmalbe = max(stmalfa, stmbeta)
            stmalbexx = max(stmalbe, stmxx)
            stminv = max(stmalbexx, 1.0)
            steg = 1.0 / stminv
            #
            xold = x.copy()
            yold = y.copy()
            zold = z.copy()
            lamold = lam.copy()
            xsiold = xsi.copy()
            etaold = eta.copy()
            muold = mu.copy()
            zetold = zet.copy()
            sold = s.copy()
            #
            itto = 0
            resinew = 2 * residunorm
            # Start: while (resinew>residunorm) and (itto<50)
            while (resinew > residunorm) and (itto < 50):
                itto = itto + 1
                x = xold + steg * dx
                y = yold + steg * dy
                z = zold + steg * dz
                lam = lamold + steg * dlam
                xsi = xsiold + steg * dxsi
                eta = etaold + steg * deta
                mu = muold + steg * dmu
                zet = zetold + steg * dzet
                s = sold + steg * ds
                ux1 = upp - x
                xl1 = x - low
                ux2 = ux1 * ux1
                xl2 = xl1 * xl1
                uxinv1 = een / ux1
                xlinv1 = een / xl1
                plam = p0 + np.dot(pp.T, lam)
                qlam = q0 + np.dot(qq.T, lam)
                gvec = np.dot(pp, uxinv1) + np.dot(qq, xlinv1)
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta
                rey = c + d * y - mu - lam
                rez = a0 - zet - np.dot(a.T, lam)
                relam = gvec - np.dot(a, z) - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn
                reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm
                rezet = np.dot(zet, z) - epsi
                res = lam * s - epsvecm
                residu1 = np.concatenate((rex, rey, rez), axis=0)
                residu2 = np.concatenate(
                    (relam, rexsi, reeta, remu, rezet, res), axis=0
                )
                residu = np.concatenate((residu1, residu2), axis=0)
                resinew = np.sqrt(np.dot(residu.T, residu))
                steg = steg / 2
                # End: while (resinew>residunorm) and (itto<50)
            residunorm = resinew.copy()
            residumax = max(abs(residu))
            steg = 2 * steg
            # End: while (residumax>0.9*epsi) and (ittt<200)
        epsi = 0.1 * epsi
        # End: while epsi>epsimin
    xmma = x.copy()
    ymma = y.copy()
    zmma = z.copy()
    lamma = lam
    xsimma = xsi
    etamma = eta
    mumma = mu
    zetmma = zet
    smma = s
    # Return values
    return xmma, ymma, zmma, lamma, xsimma, etamma, mumma, zetmma, smma


# Function for Karush–Kuhn–Tucker check
def compute_kkt_residual_on_local_approximation(
    m: int,
    n: int,
    x: ndarray,
    y: ndarray,
    z: ndarray,
    lam: ndarray,
    xsi: ndarray,
    eta: ndarray,
    mu: ndarray,
    zet: ndarray,
    s: ndarray,
    xmin: ndarray,
    xmax: ndarray,
    df0dx: ndarray,
    fval: ndarray,
    dfdx: ndarray,
    a0: float,
    a: ndarray,
    c: ndarray,
    d: ndarray,
) -> tuple[ndarray, ndarray, ndarray]:
    """KKT residual computation.

    The left hand sides of the KKT conditions for the following nonlinear programming
    problem are calculated.

    Minimize f_0(x) + a_0*z + sum(c_i*y_i + 0.5*d_i*(y_i)^2)
    subject to  f_i(x) - a_i*z - y_i <= 0,   i = 1,...,m
                xmax_j <= x_j <= xmin_j,     j = 1,...,n
                z >= 0,   y_i >= 0,          i = 1,...,m

    Args:
        m: The number of general constraints.
        n: The number of variables x_j.
        x: The current values of the n variables x_j.
        y: The current values of the m variables y_i.
        z: The current value of the single variable z.
        lam: The Lagrange multipliers for the m general constraints.
        xsi: The Lagrange multipliers for the n constraints xmin_j - x_j <= 0.
        eta: The Lagrange multipliers for the n constraints x_j - xmax_j <= 0.
        mu:  The Lagrange multipliers for the m constraints -y_i <= 0.
        zet: The Lagrange multiplier for the single constraint -z <= 0.
        s: The Slack variables for the m general constraints.
        xmin: The Lower bounds for the variables x_j.
        xmax: The Upper bounds for the variables x_j.
        df0dx: The vector with the derivatives of the objective function f_0 with
            respect to the variables x_j, calculated at x.
        fval: The vector with the values of the constraint functions f_i, calculated
        at x.
        dfdx: The (m x n)-matrix with the derivatives of the constraint functions f_i
            with respect to the variables x_j, calculated at x.
            dfdx(i,j) = the derivative of f_i with respect to x_j.
        a0: The constants a_0 in the term a_0*z.
        a: The vector with the constants a_i in the terms a_i*z.
        c: The vector with the constants c_i in the terms c_i*y_i.
        d: The vector with the constants d_i in the terms 0.5*d_i*(y_i)^2.

    Returns:
        The vector residual, its norm and its maximum.
    """
    rex = df0dx + np.dot(dfdx.T, lam) - xsi + eta
    rey = c + d * y - mu - lam
    rez = a0 - zet - np.dot(a.T, lam)
    relam = fval - a * z - y + s
    rexsi = xsi * (x - xmin)
    reeta = eta * (xmax - x)
    remu = mu * y
    rezet = zet * z
    res = lam * s
    residu1 = np.concatenate((rex, rey, rez), axis=0)
    residu2 = np.concatenate((relam, rexsi, reeta, remu, rezet, res), axis=0)
    residu = np.concatenate((residu1, residu2), axis=0)
    residunorm = np.sqrt((np.dot(residu.T, residu)).item())
    residumax = np.max(np.abs(residu))
    return residu, residunorm, residumax
