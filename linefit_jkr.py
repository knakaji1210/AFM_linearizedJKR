#!/usr/bin/env python3

"""
Linearized JKR Fitting
====

This Python script calculates the elastic modulus and adhesion energy from a 
force-distance curve, using the linearized fitting method based on the JKR 
elastic contact model.

See README.md for the details.
Command help can be shown by `python3 path/to/linefit.py -h`.

Author
----

So FUJINAMI (github ID: fujidana)

References
----

Read the following articles for the details about the method:

- "Analytical methods to derive the elastic modulus of soft and adhesive materials 
from atomic force microscopy force measurements", S. Fujinami, E. Ueda, K. Nakajima, 
T. Nishi, J. Polym. Sci. Part B Polym. Phys. 57, 1279--1286 (2019). 
DOI: 10.1002/polb.24871
- "Cone--Paraboloid Transition of the Johnson--Kendall--Roberts-Type 
Hyperboloidal Contact", S. Fujinami, K. Nakajima, Langmuir, 36, 11284--11291 (2020), 
DOI: 10.1021/acs.langmuir.0c01943

Parameters and their definitions in the code
----

- `a`: contact radius, a.
- `D`: indentation depth, D. Indentations (compressions) are positive.
- `P`: applied load, P. Repulsive forces are positive.
- `w`: work of adhesion, w.
- `K`: reduced elastic modulus, K.
"""

from __future__ import annotations
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

VERSION = '20221221'
MAX_GRAPHS = 10


def main():
    """
    main routine, which handles arguments when the program is called directly by a shell.
    """
    parser = argparse.ArgumentParser(
        description='Calculate elastic modulus from a force-distance curve by the linearized JKR fitting method.')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s {}'.format(VERSION))
    parser.add_argument('-c', '--column', action='store', type=int, nargs=2, metavar=('COL1', 'COL2'),
                        default=[0, 1], help='column indices of x and y data (0-base). Default: %(default)s, which indicates the first and second columns.')
    parser.add_argument('-s', '--skip-rows', action='store', type=int, metavar='ROWS', default=0,
                        help='skip the first ROWS lines. Use this flag if input files contain fixed number of header lines. Detault: %(default)s.')
    parser.add_argument('-f', '--flip', action='store_true',
                        help='flip data. Use this flag if the data sequence is in the counter-temporal order.')
    parser.add_argument('-o', '--offset', action='store', type=float, nargs=4, metavar=('OFF1', 'MUL1', 'OFF2', 'MUL2'),
                        help='offsets and multipilers applied to input data. Use this flag if the data is expressed not in the base SI units such as meter (m) and Newton (N).')
    parser.add_argument('-k', '--force-constant', action='store', type=float,
                        metavar='FORCE_CONST', help='normal spring constant in Newton per meter (N/m).')
    parser.add_argument('-r', '--probe-radius', action='store', type=float, required=True, metavar='RADIUS',
                        help='probe rarius in meter (m).')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-R', '--range', action='store', type=float, nargs=2, metavar=('LOWER', 'UPPER'), default=[0.3, 0.9],
                       help='lower and upper thresholds of the fitting region. The parameter should be provided as a force ratio between 0 and 1. Default value: %(default)s.')
    group.add_argument('-T', '--two-point', action='store_true',
                       help='Use the original two point method, in which the sampling points are fixed at zero-load and pull-off.')
    parser.add_argument('-g', '--graph', action='store_true',
                        help='show graph of the input data and fitting curve.')
    parser.add_argument('-G', '--graph-line', action='store_true',
                        help='show graph of the input data and fitting curve in the linearized space.')
    parser.add_argument('infiles', nargs='+', metavar='INFILE',
                        help='input data files. With -k flag, the input file should be a pair of z-scanner position and cantilever deflection. Without the flag, a pair of indentation depth and applied load.')
    args = parser.parse_args()

    v_R: float = args.probe_radius

    # read standard input if '-' is specified as the only INFILES parameter
    infiles: list[str] = []
    if len(args.infiles) == 1 and args.infiles[0] == '-':
        infiles = [line.strip() for line in sys.stdin]
    else:
        infiles = args.infiles

    # get the column width for filename in output. The minimum is 8 (since the column header is 'filename').
    colwidth_filename: int = max([len(os.path.basename(infile))
                                  for infile in infiles] + [8, ])

    if (args.graph or args.graph_line) and len(infiles) > MAX_GRAPHS:
        print('Too many input files to draw graphs. The graph option is disabled.')

    for i, infile in enumerate(infiles):
        # load X and Y waveforms from a file
        arr_x, arr_y = np.loadtxt(
            infile, usecols=args.column, unpack=True, skiprows=args.skip_rows)

        # flip the wave if the flip flag is True.
        if args.flip:
            arr_x = np.flip(arr_x)
            arr_y = np.flip(arr_y)

        if args.offset:
            if np.isnan(args.offset[0]):
                args.offset[0] = - arr_x[-1]
            if np.isnan(args.offset[2]):
                args.offset[2] = - arr_y[-1]
            arr_x = args.offset[0] + arr_x * args.offset[1]
            arr_y = args.offset[2] + arr_y * args.offset[3]

        # If a force constant is provided, the pair of waveforms should be z-scanner position and deflection.
        # Unless, the pair should be indentation dpeth and applied load.
        if args.force_constant:
            arr_D = arr_x - arr_y
            arr_P = arr_y * args.force_constant
        else:
            arr_D = arr_x
            arr_P = arr_y

        v_P_max = arr_P[0]
        # v_P_max = arr_P.max()
        pt_min = arr_P.argmin()
        v_P_min = arr_P[pt_min]

        pt_lower: float

        coef: np.ndarray
        if args.two_point:
            """traditional two-point JKR method, which uses fixed points"""
            pt_lower = pt_min
            pt_upper_float = find_level(arr_P, 0)
            pt_upper = int(np.floor(pt_upper_float))
            v_D_0 = arr_D[pt_upper] + (pt_upper_float - pt_upper) * \
                (arr_D[pt_upper + 1] - arr_D[pt_upper])
            # coef = np.polyfit(np.array([v_D_0, arr_D[pt_min]]), get_bar_D(np.array([0, v_P_min]), v_P_min), 1)
            coef = np.polyfit(np.array([v_D_0, arr_D[pt_min]]), np.array(
                [(4/3)**(1/3), -(1/12)**(1/3)]), 1)

        else:
            """new linearized JKR method"""
            if args.range[0] >= args.range[1]:
                parser.error(
                    'lower threshold of the fitting reagion must be less than upper threshold')
            elif args.range[0] < 0.0 or args.range[0] > 1.0:
                parser.error(
                    'lower threshold is beyond the range: 0 <= LOWER <= 1')
            elif args.range[1] < 0.0 or args.range[1] > 1.0:
                parser.error(
                    'upper threshold is beyond the range: 0 <= UPPER <= 1')

            # find the point indices of the upper and lower fitting thresholds.
            pt_upper = int(
                np.ceil(find_level(arr_P, (v_P_max - v_P_min) * args.range[1] + v_P_min)))
            pt_lower = int(
                np.floor(find_level(arr_P, (v_P_max - v_P_min) * args.range[0] + v_P_min)))
            # do line fitting.
            if pt_upper > pt_lower:
                raise Exception('Failed in finding range. Upper threshold (pt:{}) comes after lower threshold (pt:{}).'.format(pt_upper, pt_lower))
            coef = np.polyfit(
                arr_D[pt_upper:pt_lower+1], get_bar_D(arr_P[pt_upper:pt_lower + 1], v_P_min), 1)

        v_w = - 2 * v_P_min / (3 * np.pi * v_R)
        v_K = - 2 * v_P_min / 3 * np.sqrt(coef[0] ** 3 / v_R)
        intercept_D = - coef[1] / coef[0]

        if i == 0:
            if len(infiles) > 1:
                print(('{:%ds} ' % colwidth_filename).format(
                    'filename'), end='')
            print('{:12s} {:12s}'.format('K [Pa]', 'w [J/m2]'))

        if len(infiles) > 1:
            print(('{:%ds} ' % colwidth_filename).format(
                os.path.basename(infile)), end='')
        print('{:12.6g} {:12.6g}'.format(v_K, v_w))

        if (args.graph or args.graph_line) and len(infiles) <= MAX_GRAPHS:
            if args.graph_line:
                arr_D_2pt = np.array([arr_D[pt_lower], arr_D[pt_upper]])
                plt.plot(arr_D, get_bar_D(arr_P, v_P_min),
                         '.', label='data', color='red')
                plt.plot(arr_D[0:pt_min + 1], coef[0] * arr_D[0:pt_min + 1] + coef[1],
                         ':', label='line fitting  (extended)', color='blue')
                plt.plot(arr_D_2pt, coef[0] * arr_D_2pt + coef[1],
                         '-', label='line fitting (fit range only)', color='blue')
                plt.legend()
                plt.show()

            if args.graph:
                nondim_coef = get_nondim_coef(v_K, v_w, v_R)
                fit_part = calc_jkr_curve(
                    nondim_coef, arr_P[pt_lower], arr_P[pt_upper])
                # fit_full = calc_JKR_curve(nondim_coef, arr_P[pt_min], arr_P[0])
                fit_full = calc_jkr_curve(nondim_coef, np.nan, arr_P[0])

                plt.plot(arr_D, arr_P, '.', label='data', color='red')
                plt.plot(fit_full[1] + intercept_D, fit_full[2],
                         ':', label='linearized fitting (extended)', color='blue')
                plt.plot(fit_part[1] + intercept_D, fit_part[2], '-',
                         label='linearized fitting (fit range only)', color='blue')
                plt.legend()
                plt.show()


def find_level(arr_in: np.ndarray, level: float) -> float:
    """
    Get the index of 1-D numpy array `arr_in` in which the value crosses the `level`.
    Return nan if not found.
    """
    if arr_in[0] == level:
        return 0.0
    elif arr_in[0] > level:
        indices = np.flatnonzero(arr_in <= level)
    else:
        indices = np.flatnonzero(arr_in >= level)

    if indices.size == 0:
        return np.nan
    else:
        ind = indices[0]
        # return ind

        # return an interpolated index (floating point number)
        return ind - (arr_in[ind] - float(level)) / (arr_in[ind] - arr_in[ind - 1])


def get_bar_D(v_P: float | np.ndarray, v_P_c: float) -> float | np.ndarray:
    """
    Calculate the nondimesional indentation depth, bar_D.

    This is a preprocess in the linearized JKR fitting method.
    """
    norm_P = - v_P / v_P_c
    sqrt_part = np.sqrt(norm_P + 1)
    return (1.5 * norm_P + 1 + sqrt_part) / ((1.5 * norm_P + 3 + 3 * sqrt_part) ** (1/3))


def get_nondim_coef(v_K: float, v_w: float, v_R: float) -> tuple[float, float, float]:
    """
    Calculate the coefficients of Maugis nondimensional parameters.
    """
    coef_P = np.pi * v_w * v_R
    coef_a = (coef_P * v_R / v_K) ** (1/3)
    coef_D = (coef_a * coef_a / v_R)
    return coef_a, coef_D, coef_P


def get_bar_a(v_bar_P: float) -> float:
    """
    Calculate the nondimensional contact radius, bar_a, from the nondimensional load, bar_P.
    """
    return (v_bar_P + 3 + np.sqrt(6 * v_bar_P + 9)) ** (1/3)


def calc_jkr_curve(nondim_coef: tuple[float, float, float], v_P_min: float, v_P_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate a theoretical JKR curve between the specified load range.
    If v_P_min is NaN, the curve starts from the origin (a = 0).
    """
    v_bar_a_min = get_bar_a(
        v_P_min / nondim_coef[2]) if not np.isnan(v_P_min) else 0.0
    v_bar_a_max = get_bar_a(v_P_max / nondim_coef[2])

    arr_fit_bar_a = np.linspace(v_bar_a_min, v_bar_a_max, 100, dtype=float)
    arr_fit_D = nondim_coef[1] * (arr_fit_bar_a **
                                  2 - 2 / 3 * np.sqrt(6 * arr_fit_bar_a))
    arr_fit_P = nondim_coef[2] * (arr_fit_bar_a **
                                  3 - np.sqrt(6 * (arr_fit_bar_a ** 3)))
    return nondim_coef[0] * arr_fit_bar_a, arr_fit_D, arr_fit_P


if __name__ == '__main__':
    main()