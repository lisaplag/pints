#
# A collection of useful functions when dealing with emulators
#

from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals

import numpy as np


def fix_parameters(bounds):
    """
    Returns lists consisting of (parameter index, value) tuples.
    Use for visualisations in situations where there are more than 2
    inputs to a model
    """
    n_parameters = bounds.n_parameters()
    fixed_parameters = []
    for i in range(n_parameters):
        for j in range(n_parameters):
            if i == j:
                continue
            mid_vals = enumerate((bounds.lower()+bounds.upper())/2)
            mid_vals = list(mid_vals)
            mid_vals = [(idx, val) for (idx, val) in mid_vals
                        if idx not in [i, j]]

            fixed_parameters.append(mid_vals)

    return fixed_parameters


def generate_grid(lower, upper, splits, fixed=[]):
    """
    Generates a grid of evenly spaced out points for testing
    returns grid of the first paramater, grid of second parameter,
    and their values stacked with any fixed values provided
    Generated values are convenient for plotting surface
    or contour plots.
    fixed -- contains position and argument to keep fixed
    """

    if not fixed:
        p1_low, p2_low = lower
        p1_high, p2_high = upper
    else:
        # find out which parameters are not fixed and get their bounds
        n_params = len(lower)

        p1_idx, p2_idx = [i for i in range(n_params)
                          if i not in [j for (j, _) in fixed]]

        p1_low, p2_low = lower[p1_idx], lower[p2_idx]
        p1_high, p2_high = upper[p1_idx], upper[p2_idx]

    p1_range = np.linspace(p1_low, p1_high, splits)
    p2_range = np.linspace(p2_low, p2_high, splits)
    p1_grid, p2_grid = np.meshgrid(p1_range, p2_range)

    if fixed:
        # create a grid for every parameter and insert in
        # corresponding position in the grids array
        grids = [None] * n_params
        grids[p1_idx] = p1_grid
        grids[p2_idx] = p2_grid
        for (i, val) in fixed:
            fixed_grid = np.zeros((splits, splits)) + val
            grids[i] = fixed_grid
    else:
        grids = [p1_grid, p2_grid]

    grid = np.dstack(tuple(grids))

    return p1_grid, p2_grid, grid


def predict_grid(model, grid, dims=None):
    """
    Given a PDF and a grid of inputs calculates probability for
    each index in the grid
    """
    rows, cols, n_params = grid.shape
    flatten_grid = grid.reshape((rows * cols, n_params))
    pred = np.apply_along_axis(model, 1, flatten_grid)
    return pred.reshape(rows, cols)


# Functions to deal with composite kernels
def is_prod_kernel(kernel):
    """
    True when a given kernel is a GPy Prod kernel
    """
    return type(kernel) == kern.src.prod.Prod


def is_add_kernel(kernel):
    """
    True when a given kernel is a GPy Add kernel
    """
    return type(kernel) == kern.src.add.Add


def kernel_to_string(kernel, ident=0, decimal_places=4):
    """
    Converts complex GPy kernels to strings
    TODO: rewrite with .format
    """
    if kernel is None:
        return ""
    s = ""
    formatting = "{:." + str(decimal_places) + "f}"
    tab = ident * " "
    if is_prod_kernel(kernel) or is_add_kernel(kernel):
        op = "*" if is_prod_kernel(kernel) else "+"
        sub_kernels = []
        for sub_kernel in kernel.parameters:
            sub_kernels.append(kernel_to_string(sub_kernel, ident=ident + 1))
        s = "(" + op + "\n" + "\n".join(sub_kernels) + "\n" + tab + ")"
    else:
        # get name of kernel without "'>" characters
        name = str(type(kernel)).split(".")[-1]
        name = name[:-2]

        values = ",".join([formatting.format(x) for x in kernel])
        s = name + "(" + values + ")"
    return " " * ident + s


def get_total_variance(kernel):
    ans = 0
    if is_prod_kernel(kernel) or is_add_kernel(kernel):
        for sub_kernel in kernel.parameters:
            ans += get_total_variance(sub_kernel)
    else:
        if hasattr(kernel, "variance"):
            ans += kernel.variance
        elif hasattr(kernel, "variances"):
            ans += kernel.variances
        else:
            ans += 0

    return ans


def has_high_variance(kernel, threshold=10):
    variance = 0
    if is_prod_kernel(kernel) or is_add_kernel(kernel):
        for sub_kernel in kernel.parameters:
            if has_high_variance(sub_kernel):
                return True
    else:
        if hasattr(kernel, "variances"):
            variance = kernel.variances
        elif hasattr(kernel, "variance"):
            variance = kernel.variance
        else:
            # some kernels don't have variance as a parameter
            return False

    return variance > threshold


def simulate(
    model,
    parameters=None,
    times=None,
    noise_range_percent=0.05,
    n_splits=None
):
    """
    Simulates model for specified time interval with specified noise.
    Noise is normal with standart deviation as: range * noise_range_percent.
    Pass noise_range_percent=None if no noise wanted
    Returns values, times, noise_stds
    If n_splits is provided divide time interval into n_splits uniform parts.
    """

    if parameters is None:
        parameters = np.array(model.suggested_parameters())

    if times is None:
        times = model.suggested_times()

    # take times and calculate values
    if n_splits:
        min_time, max_time = min(times), max(times)
        times = np.linspace(min_time, max_time, n_splits)

    # simulate
    values = model.simulate(parameters, times)

    # noise
    # by default set 5% of range as the standard deviation
    if noise_range_percent is not None:
        noise_stds = np.abs(values.max(axis=0) - values.min(axis=0)) * noise_range_percent

        # final values
        values = values + np.random.normal(0, noise_stds, values.shape)

        return values, times, noise_stds

    return values, times