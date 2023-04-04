import math


def pretty(start: float, stop: float, num: int = 5) -> list:
    """Pretty Intervals.

    Returns about num evenly spaced values covering the interval [start, stop].
    In contrast to `numpy.linspace()`, the values will be "pretty", see
    https://stackoverflow.com/questions/43075617/python-function-equivalent-to-rs-pretty

    Parameters
    ----------
    start : float
        Lowest value to be covered.
    stop : float
        Highest value to be covered.
    num : int
        Approximate number of bins.

    Returns
    -------
    list

    Examples
    --------
    >>> pretty(-1, 101)
    [-20, 0, 20, 40, 60, 80, 100, 120]

    >>> pretty(0, 0.1, num=6)
    [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    """
    cell = (stop - start) / num
    base = 10 ** math.floor(math.log10(cell))
    ratio = cell / base
    if ratio <= 1.4:
        c = 1
    elif ratio <= 2.8:
        c = 2
    elif ratio <= 7:
        c = 5
    else:
        c = 10
    unit = c * base
    ns = math.floor(start / unit + 1e-10)
    nu = math.ceil(stop / unit - 1e-10)
    breaks = [unit * i for i in range(ns, nu + 1)]

    return breaks
