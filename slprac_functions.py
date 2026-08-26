# This file contains helper functions for the sea level practical. Understanding
# these functions is not critical for understanding the sea level practical, but
# they are provided here for your reference.

from pathlib import Path
from scipy.io import loadmat
from scipy.stats import linregress, t as student_t
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

def locate_practical_file(relative_path):
    """Find a file in the original practical directory layout."""
    relative_path = Path(relative_path)
    candidates = [
        Path.cwd() / relative_path,
        Path.cwd() / "SLprac" / relative_path,
        Path.cwd().parent / "SLprac" / relative_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {relative_path}. Searched:\n{searched}")


def matlab_datenum_to_datetime(datenum):
    """Convert MATLAB serial datenums to a pandas DatetimeIndex."""
    datenum = np.asarray(datenum, dtype=float)
    # MATLAB datenum 719529 corresponds to 1970-01-01.
    return pd.DatetimeIndex(
        pd.to_datetime(datenum - 719529.0, unit="D", origin="unix")
    ).round("s")


def load_tide_gauge(path):
    """Load the supplied WLData matrix and return an hourly pandas Series."""
    mat = loadmat(path)
    wl = np.asarray(mat["WLData"], dtype=float)
    time = matlab_datenum_to_datetime(wl[:, 0]).round("h")
    series = pd.Series(wl[:, 1], index=time, name="sea_level_m").sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return series


def load_bats_ts(path):
    """Load the supplied MATLAB TS structure into NumPy arrays."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    ts = mat["TS"]
    return {
        "time": matlab_datenum_to_datetime(ts.Time),
        "temperature": np.asarray(ts.Temp, dtype=float),
        "salinity": np.asarray(ts.Sal, dtype=float),
        "pressure": np.asarray(ts.Pres, dtype=float),
    }


def format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def decimal_year(index):
    """Convert datetime values to decimal years for regression."""
    index = pd.DatetimeIndex(index)
    year_start = pd.to_datetime(index.year.astype(str) + "-01-01")
    next_start = pd.to_datetime((index.year + 1).astype(str) + "-01-01")
    return index.year + (index - year_start) / (next_start - year_start)

def interval_mean(series, freq="YS", min_coverage=0.50):
    """
    Calculate means over arbitrary calendar intervals.

    Parameters
    ----------
    series : pandas.Series
        Regularly sampled time series.
    freq : str
        Pandas resampling frequency, e.g. "YS", "MS", "DS".
    min_coverage : float
        Minimum fraction of expected observations required.

    Returns
    -------
    means, coverage : pandas.Series
        Interval means and corresponding coverage fractions.
    """
    means = series.resample(freq).mean()
    counts = series.resample(freq).count()

    dt = series.index.to_series().diff().median()

    starts = means.index
    ends = starts + pd.tseries.frequencies.to_offset(freq)

    expected = (ends - starts) / dt
    coverage = counts / expected

    return means.where(coverage >= min_coverage), coverage

def fit_linear_trend(annual_series, start=None, end=None, confidence=0.95):
    """
    Fit y = intercept + slope * year to annual means.

    The confidence interval is the conventional Student-t interval on the OLS
    slope. Annual averaging greatly reduces tidal/weather autocorrelation, but
    the interval still assumes independent regression residuals.
    """
    data = annual_series.dropna().copy()
    if start is not None:
        data = data.loc[pd.Timestamp(start):]
    if end is not None:
        data = data.loc[:pd.Timestamp(end)]
    if len(data) < 3:
        raise ValueError("At least three annual means are required for a trend.")

    x = np.asarray(decimal_year(data.index), dtype=float)
    y = data.to_numpy(dtype=float)
    result = linregress(x, y)
    dof = len(data) - 2
    critical = student_t.ppf(0.5 + confidence / 2, dof)
    slope_half_width = critical * result.stderr

    fitted = pd.Series(result.intercept + result.slope * x, index=data.index)
    return {
        "data": data,
        "fitted": fitted,
        "slope_m_per_year": result.slope,
        "slope_mm_per_year": result.slope * 1000,
        "slope_ci_half_mm_per_year": slope_half_width * 1000,
        "intercept": result.intercept,
        "r_squared": result.rvalue ** 2,
        "n_years": len(data),
        "confidence": confidence,
    }