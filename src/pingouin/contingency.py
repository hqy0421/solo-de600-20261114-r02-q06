# Date: May 2019
import itertools
import numbers
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binom, power_divergence
from scipy.stats import chi2 as sp_chi2
from scipy.stats.contingency import expected_freq

from .power import power_chi2
from .utils import _postprocess_dataframe, _register_dataframe_method

__all__ = ["chi2_independence", "chi2_mcnemar", "dichotomous_crosstab", "loglinear"]


###############################################################################
# CHI-SQUARED TESTS
###############################################################################


@_register_dataframe_method
def chi2_independence(data, x, y, correction=True):
    """
    Chi-squared independence tests between two categorical variables.

    The test is computed for different values of :math:`\\lambda`: 1, 2/3, 0,
    -1/2, -1 and -2 (Cressie and Read, 1984).

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        The dataframe containing the ocurrences for the test.
    x, y : string
        The variables names for the Chi-squared test. Must be names of columns
        in ``data``.
    correction : bool
        Whether to apply Yates' correction when the degree of freedom of the
        observed contingency table is 1 (Yates 1934).

    Returns
    -------
    expected : :py:class:`pandas.DataFrame`
        The expected contingency table of frequencies.
    observed : :py:class:`pandas.DataFrame`
        The (corrected or not) observed contingency table of frequencies.
    stats : :py:class:`pandas.DataFrame`
        The test summary, containing four columns:

        * ``'test'``: The statistic name
        * ``'lambda'``: The :math:`\\lambda` value used for the power\
                        divergence statistic
        * ``'chi2'``: The test statistic
        * ``'pval'``: The p-value of the test
        * ``'cramer'``: The Cramer's V effect size
        * ``'power'``: The statistical power of the test

    Notes
    -----
    From Wikipedia:

    *The chi-squared test is used to determine whether there is a significant
    difference between the expected frequencies and the observed frequencies
    in one or more categories.*

    As application examples, this test can be used to *i*) evaluate the
    quality of a categorical variable in a classification problem or to *ii*)
    check the similarity between two categorical variables. In the first
    example, a good categorical predictor and the class column should present
    high :math:`\\chi^2` and low p-value. In the second example, similar
    categorical variables should present low :math:`\\chi^2` and high p-value.

    This function is a wrapper around the
    :py:func:`scipy.stats.power_divergence` function.

    .. warning :: As a general guideline for the consistency of this test, the
        observed and the expected contingency tables should not have cells
        with frequencies lower than 5.

    References
    ----------
    .. [1] Cressie, N., & Read, T. R. (1984). Multinomial goodness‐of‐fit
           tests. Journal of the Royal Statistical Society: Series B
           (Methodological), 46(3), 440-464.

    .. [2] Yates, F. (1934). Contingency Tables Involving Small Numbers and the
           :math:`\\chi^2` Test. Supplement to the Journal of the Royal
           Statistical Society, 1, 217-235.

    Examples
    --------
    Let's see if gender is a good categorical predictor for the presence of
    heart disease.

    >>> import pingouin as pg
    >>> data = pg.read_dataset('chi2_independence')
    >>> data['sex'].value_counts(ascending=True)
    sex
    0     96
    1    207
    Name: count, dtype: int64

    If gender is not a good predictor for heart disease, we should expect the
    same 96:207 ratio across the target classes.

    >>> expected, observed, stats = pg.chi2_independence(data, x='sex',
    ...                                                  y='target')
    >>> expected
    target          0           1
    sex
    0       43.722772   52.277228
    1       94.277228  112.722772

    Let's see what the data tells us.

    >>> observed
    target      0     1
    sex
    0        24.5  71.5
    1       113.5  93.5

    The proportion is lower on the class 0 and higher on the class 1. The
    tests should be sensitive to this difference.

    >>> stats.round(3)
                     test  lambda    chi2  dof  pval  cramer  power
    0             pearson   1.000  22.717  1.0   0.0   0.274  0.997
    1        cressie-read   0.667  22.931  1.0   0.0   0.275  0.998
    2      log-likelihood   0.000  23.557  1.0   0.0   0.279  0.998
    3       freeman-tukey  -0.500  24.220  1.0   0.0   0.283  0.998
    4  mod-log-likelihood  -1.000  25.071  1.0   0.0   0.288  0.999
    5              neyman  -2.000  27.458  1.0   0.0   0.301  0.999

    Very low p-values indeed. The gender qualifies as a good predictor for the
    presence of heart disease on this dataset.
    """
    # Python code inspired by SciPy's chi2_contingency
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be a pandas DataFrame, but received {type(data).__name__}.")
    for name, col in (("x", x), ("y", y)):
        if not isinstance(col, (str, int)):
            raise TypeError(f"'{name}' must be a string or int, but received {type(col).__name__}.")
    if not all(col in data.columns for col in (x, y)):
        raise ValueError("columns are not in dataframe.")
    if not isinstance(correction, bool):
        raise TypeError(
            f"'correction' must be a boolean, but received {type(correction).__name__}."
        )

    observed = pd.crosstab(data[x], data[y])

    if observed.size == 0:
        raise ValueError("No data; observed has size 0.")

    expected = pd.DataFrame(expected_freq(observed), index=observed.index, columns=observed.columns)

    # All count frequencies should be at least 5
    for df, name in zip([observed, expected], ["observed", "expected"], strict=True):
        if (df < 5).any(axis=None):
            warnings.warn(f"Low count on {name} frequencies.", stacklevel=2)

    dof = float(expected.size - sum(expected.shape) + expected.ndim - 1)

    # Number of observations in the table (rows with missing values are dropped by crosstab)
    n = observed.to_numpy().sum()

    if dof == 1 and correction:
        # Adjust `observed` according to Yates' correction for continuity. Same as
        # scipy.stats.chi2_contingency, the correction cannot exceed the observed deviation.
        diff = expected - observed
        observed = observed + np.sign(diff) * np.minimum(0.5, np.abs(diff))

    ddof = observed.size - 1 - dof
    dof_cramer = min(expected.shape) - 1
    tests = {
        "pearson": 1.0,
        "cressie-read": 2 / 3,
        "log-likelihood": 0.0,
        "freeman-tukey": -1 / 2,
        "mod-log-likelihood": -1.0,
        "neyman": -2.0,
    }
    stats = []
    for name, lambda_ in tests.items():
        if dof == 0:
            chi2, p, cramer, power = 0.0, 1.0, np.nan, np.nan
        else:
            chi2, p = power_divergence(
                observed.to_numpy(), expected.to_numpy(), ddof=ddof, axis=None, lambda_=lambda_
            )
            cramer = np.sqrt(chi2 / (n * dof_cramer))
            power = power_chi2(dof=dof, w=cramer, n=n, alpha=0.05)
        stats.append(
            {
                "test": name,
                "lambda": lambda_,
                "chi2": chi2,
                "dof": dof,
                "pval": p,
                "cramer": cramer,
                "power": power,
            }
        )

    return expected, observed, _postprocess_dataframe(pd.DataFrame(stats))


@_register_dataframe_method
def chi2_mcnemar(data, x, y, correction=True):
    """
    Performs the exact and approximated versions of McNemar's test.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        The dataframe containing the ocurrences for the test. Each row must
        represent either a subject or a pair of subjects.
    x, y : string
        The variables names for the McNemar's test. Must be names of columns
        in ``data``.

        If each row of ``data`` represents a subject, then ``x`` and ``y`` must
        be columns containing dichotomous measurements in two different
        contexts. For instance: the presence of pain before and after a certain
        treatment.

        If each row of ``data`` represents a pair of subjects, then ``x`` and
        ``y`` must be columns containing dichotomous measurements for each of
        the subjects. For instance: a positive response to a certain drug in
        the control group and in the test group, supposing that each pair
        contains a subject in each group.

        The 2x2 crosstab is created using the
        :py:func:`pingouin.dichotomous_crosstab` function.

        .. warning:: Missing values are not allowed.

    correction : bool
        Whether to apply the correction for continuity (Edwards, A. 1948).

    Returns
    -------
    observed : :py:class:`pandas.DataFrame`
        The observed contingency table of frequencies.
    stats : :py:class:`pandas.DataFrame`
        The test summary:

        * ``'chi2'``: The test statistic
        * ``'dof'``: The degree of freedom
        * ``'p_approx'``: The approximated p-value
        * ``'p_exact'``: The exact p-value

    Notes
    -----
    The McNemar's test is compatible with dichotomous paired data, generally
    used to assert the effectiveness of a certain procedure, such as a
    treatment or the use of a drug. "Dichotomous" means that the values of the
    measurements are binary. "Paired data" means that each measurement is done
    twice, either on the same subject in two different moments or in two
    similar (paired) subjects from different groups (e.g.: control/test). In
    order to better understand the idea behind McNemar's test, let's illustrate
    it with an example.

    Suppose that we wanted to compare the effectiveness of two different
    treatments (X and Y) for athlete's foot on a certain group of `n` people.
    To achieve this, we measured their responses to such treatments on each
    foot. The observed data summary was:

    * Number of people with good responses to X and Y: `a`
    * Number of people with good response to X and bad response to Y: `b`
    * Number of people with bad response to X and good response to Y: `c`
    * Number of people with bad responses to X and Y: `d`

    Now consider the two groups:

    1. The group of people who had good response to X (`a` + `b` subjects)
    2. The group of people who had good response to Y (`a` + `c` subjects)

    If the treatments have the same effectiveness, we should expect the
    probabilities of having good responses to be the same, regardless of the
    treatment. Mathematically, such statement can be translated into the
    following equation:

    .. math::

        \\frac{a+b}{n} = \\frac{a+c}{n} \\Rightarrow b = c

    Thus, this test should indicate higher statistical significances for higher
    distances between `b` and `c` (McNemar, Q. 1947):

    .. math::

        \\chi^2 = \\frac{(b - c)^2}{b + c}

    References
    ----------
    .. [1] Edwards, A. L. (1948). Note on the "correction for continuity" in
           testing the significance of the difference between correlated
           proportions. Psychometrika, 13(3), 185-187.

    .. [2] McNemar, Q. (1947). Note on the sampling error of the difference
           between correlated proportions or percentages. Psychometrika, 12(2),
           153-157.

    Examples
    --------
    >>> import pingouin as pg
    >>> data = pg.read_dataset("chi2_mcnemar")
    >>> observed, stats = pg.chi2_mcnemar(data, "treatment_X", "treatment_Y")
    >>> observed
    treatment_Y   0   1
    treatment_X
    0            20  40
    1             8  12

    In this case, `c` (40) seems to be a significantly greater than `b` (8).
    The McNemar test should be sensitive to this.

    >>> stats
                chi2  dof  p_approx   p_exact
    mcnemar  20.020833    1  0.000008  0.000003
    """
    # Python code initially inspired by statsmodel's mcnemar
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be a pandas DataFrame, but received {type(data).__name__}.")
    for name, col in (("x", x), ("y", y)):
        if not isinstance(col, (str, int)):
            raise TypeError(f"'{name}' must be a string or int, but received {type(col).__name__}.")
    if not all(column in data.columns for column in (x, y)):
        raise ValueError("columns are not in dataframe.")

    for column in (x, y):
        if data[column].isna().any():
            raise ValueError("Null values are not allowed.")

    observed = dichotomous_crosstab(data, x, y)
    # Careful, the order of b and c is inverted compared to wikipedia
    # because the colums / rows of the crosstab is [0, 1] and not [1, 0].
    c, b = observed.at[0, 1], observed.at[1, 0]
    n_discordants = b + c

    if (b, c) == (0, 0):
        raise ValueError(
            "McNemar's test does not work if the secondary "
            "diagonal of the observed data summary does not "
            "have values different from 0."
        )

    chi2 = (abs(b - c) - int(correction)) ** 2 / n_discordants
    pexact = min(1, 2 * binom.cdf(min(b, c), n_discordants, 0.5))
    stats = {
        "chi2": chi2,
        "dof": 1,
        "p_approx": sp_chi2.sf(chi2, 1),
        "p_exact": pexact,
    }

    stats = pd.DataFrame(stats, index=["mcnemar"])

    return observed, _postprocess_dataframe(stats)


###############################################################################
# DICHOTOMOUS CONTINGENCY TABLES
###############################################################################


def _dichotomize_series(data, column):
    """Converts the values of a pd.DataFrame column into 0 or 1"""
    series = data[column]
    if series.dtype == bool:
        return series.astype(int)

    def convert_elem(elem):
        if isinstance(elem, (numbers.Real, np.bool_)) and elem in (0, 1):
            return int(elem)
        if isinstance(elem, str):
            lower = elem.lower()
            if lower in ("n", "no", "absent", "false", "f", "negative"):
                return 0
            elif lower in ("y", "yes", "present", "true", "t", "positive", "p"):
                return 1
        raise ValueError(
            f"Invalid value to build a 2x2 contingency table on column {column}: {elem}"
        )

    # Convert each unique value only once
    return series.map({elem: convert_elem(elem) for elem in series.unique()})


@_register_dataframe_method
def dichotomous_crosstab(data, x, y):
    """
    Generates a 2x2 contingency table from a :py:class:`pandas.DataFrame` that
    contains only dichotomous entries, which are converted to 0 or 1.

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        Pandas dataframe
    x, y : string
        Column names in ``data``.

        Currently, Pingouin recognizes the following values as dichotomous
        measurements:

        * ``0``, ``0.0``, ``False``, ``'No'``, ``'N'``, ``'Absent'``,\
        ``'False'``, ``'F'`` or ``'Negative'`` for negative cases;

        * ``1``, ``1.0``, ``True``, ``'Yes'``, ``'Y'``, ``'Present'``,\
        ``'True'``, ``'T'``, ``'Positive'`` or ``'P'``,  for positive cases;

        If strings are used, Pingouin will recognize them regardless of their
        uppercase/lowercase combinations.

    Returns
    -------
    crosstab : :py:class:`pandas.DataFrame`
        The 2x2 crosstab. See :py:func:`pandas.crosstab` for more details.

    Examples
    --------
    >>> import pandas as pd
    >>> import pingouin as pg
    >>> df = pd.DataFrame({'A': ['Yes', 'No', 'No'], 'B': [0., 1., 0.]})
    >>> pg.dichotomous_crosstab(data=df, x='A', y='B')
    B  0  1
    A
    0  1  1
    1  1  0
    """
    crosstab = pd.crosstab(_dichotomize_series(data, x), _dichotomize_series(data, y))
    # Add the missing level(s) with a count of zero, e.g. if all the values of x are 1
    return crosstab.reindex(index=[0, 1], columns=[0, 1], fill_value=0)


###############################################################################
# LOG-LINEAR MODELS FOR MULTIWAY CONTINGENCY TABLES
###############################################################################


def _loglinear_generators(x, interactions):
    """Maximal interaction terms (generators) of the hierarchical log-linear model.

    Main effects are always included and any interaction implies all of its
    lower-order subsets, so the sufficient margins of the model are exactly the
    margins defined by the returned maximal terms.
    """
    gens = {frozenset([col]) for col in x}  # main effects
    for term in interactions:
        if not isinstance(term, (list, tuple, set, frozenset)):
            raise TypeError("Each interaction term must be a list, tuple or set of column names.")
        term = frozenset(term)
        if len(term) == 0:
            raise ValueError("Interaction terms cannot be empty.")
        if not term.issubset(set(x)):
            raise ValueError("All variables in 'interactions' must be columns of 'x'.")
        gens.add(term)
    # Keep only the maximal terms: a term contained in another one is redundant
    maximal = [g for g in gens if not any(g < h for h in gens)]
    # Deterministic order: by size, then alphabetically
    return sorted(maximal, key=lambda g: (len(g), sorted(str(c) for c in g)))


def _loglinear_ipf(obs, allowed, gen_buckets, gen_sizes, max_iter=1000, tol=1e-8):
    """Iterative proportional fitting (IPF) of the expected frequencies.

    Structural-zero cells (``allowed == False``) are fixed at zero and excluded
    from the fit: the model margins are matched on the remaining cells only.
    Returns the fitted expected frequencies (flattened, same order as ``obs``).
    """
    expected = np.where(allowed, 1.0, 0.0)
    threshold = tol * max(obs.sum(), 1.0)
    for _ in range(max_iter):
        converged = True
        for bucket, size in zip(gen_buckets, gen_sizes, strict=True):
            obs_margin = np.bincount(bucket, weights=obs, minlength=size)
            exp_margin = np.bincount(bucket, weights=expected, minlength=size)
            if np.abs(exp_margin - obs_margin).max() > threshold:
                converged = False
            null = exp_margin == 0
            if np.any(null & (obs_margin > 0)):
                raise ValueError(
                    "The log-linear model cannot reproduce the observed margins: a "
                    "margin that is zero under the model has a positive observed "
                    "count. This is usually caused by structural zeros that are "
                    "incompatible with the requested interactions."
                )
            ratio = np.ones(size)
            ratio[~null] = obs_margin[~null] / exp_margin[~null]
            # Structural-zero cells stay at exactly zero (0 * ratio = 0)
            expected = expected * ratio[bucket]
        if converged:
            return expected
    warnings.warn(
        f"IPF did not converge after {max_iter} iterations; the expected "
        "frequencies may be inaccurate.",
        stacklevel=3,
    )
    return expected


def _loglinear_mcmc(obs, expected, allowed, shape, gen_sets, gen_buckets, n_mcmc, burn_in, rng):
    """Metropolis-Hastings sampling of tables with fixed sufficient margins.

    Samples tables from the exact conditional distribution of the multiway
    contingency table given the sufficient margins of the fitted log-linear
    model. Under Poisson, multinomial or product-multinomial sampling, this
    distribution is proportional to :math:`1 / \\prod_c x_c!` over the tables
    that share the observed sufficient margins, and does not depend on the
    model parameters. Structural-zero cells are kept at zero throughout.

    Two kinds of margin-preserving moves are proposed (Diaconis & Sturmfels
    1998): "basic" moves that swap one count within a 2 x 2 sub-table of two
    factors, and full 2^d "checkerboard" moves that add +-1 in alternation on
    the corners of a sub-box of the table. Every proposed move is additionally
    checked numerically against all sufficient margins, so the stationary
    distribution is correct by construction.

    Parameters
    ----------
    obs : 1D array
        Observed counts (flattened table), the starting state of the chain.
    expected : 1D array
        Fitted expected frequencies (flattened). Constant over the fiber, since
        all tables in the fiber share the sufficient margins.
    allowed : 1D bool array
        False for structural-zero cells.
    shape : tuple of int
        Shape of the contingency table.
    gen_sets : list of frozenset
        Factor positions (0-based) of each generator of the model.
    gen_buckets : list of 1D int arrays
        For each generator, the margin bucket of each (flattened) cell.
    n_mcmc, burn_in : int
        Number of kept samples and of discarded burn-in steps.
    rng : :py:class:`numpy.random.Generator`

    Returns
    -------
    stats : (n_mcmc, 2) array
        Pearson X2 and likelihood-ratio G2 of each sampled table.
    last : 1D array
        The last sampled table (used to check margin preservation in tests).
    """
    ndim = len(shape)
    # Basic moves on a factor pair preserve every margin except those whose
    # factor set contains the pair; precompute the allowed pairs. Factors with
    # a single level cannot take part in a move.
    pairs = [
        (i, j)
        for i in range(ndim)
        for j in range(i + 1, ndim)
        if shape[i] >= 2 and shape[j] >= 2 and not any({i, j} <= g for g in gen_sets)
    ]
    # A checkerboard move over all factors preserves every proper margin; it is
    # useless for a saturated model or if a factor has a single level.
    use_checkerboard = not any(len(g) == ndim for g in gen_sets) and min(shape) >= 2
    if not pairs and not use_checkerboard:
        return None, obs.copy()

    def propose():
        """Sample a margin-preserving move: cells (flat indices) and +-1 deltas."""
        if pairs and (not use_checkerboard or rng.random() < 0.5):
            i, j = pairs[rng.integers(len(pairs))]
            li = rng.choice(shape[i], size=2, replace=False)
            lj = rng.choice(shape[j], size=2, replace=False)
            base = [rng.integers(s) for s in shape]
            cells, deltas = [], []
            for a, b, delta in ((0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)):
                coord = base.copy()
                coord[i], coord[j] = li[a], lj[b]
                cells.append(np.ravel_multi_index(coord, shape))
                deltas.append(delta)
        else:
            levels = [rng.choice(s, size=2, replace=False) for s in shape]
            cells, deltas = [], []
            for combo in itertools.product((0, 1), repeat=ndim):
                cells.append(
                    np.ravel_multi_index([levels[k][combo[k]] for k in range(ndim)], shape)
                )
                deltas.append(1 if sum(combo) % 2 == 0 else -1)
        return np.asarray(cells), np.asarray(deltas)

    def preserves_margins(cells, deltas):
        """Numerical check that a move leaves every sufficient margin unchanged."""
        for bucket in gen_buckets:
            if np.any(np.bincount(bucket[cells], weights=deltas) != 0):
                return False
        return True

    state = obs.copy()
    pos = expected > 0
    x2 = np.sum((state[pos] - expected[pos]) ** 2 / expected[pos])
    g2 = 2 * np.sum(
        state[pos & (state > 0)] * np.log(state[pos & (state > 0)] / expected[pos & (state > 0)])
    )
    stats = np.empty((n_mcmc, 2))
    for step in range(burn_in + n_mcmc):
        cells, deltas = propose()
        # Reject (stay) if the move hits a structural zero, a negative count,
        # or changes a sufficient margin.
        if allowed[cells].all() and (state[cells] + deltas >= 0).all():
            if preserves_margins(cells, deltas):
                # Log of the Metropolis-Hastings ratio (symmetric proposal):
                # pi(x) proportional to 1 / prod(x_c!)
                log_alpha = 0.0
                for c, d in zip(cells, deltas, strict=True):
                    log_alpha += -np.log(state[c] + 1) if d == 1 else np.log(state[c])
                if log_alpha >= 0 or np.log(rng.random()) < log_alpha:
                    for c, d in zip(cells, deltas, strict=True):
                        old, e = state[c], expected[c]
                        if e > 0:
                            x2 -= (old - e) ** 2 / e
                            if old > 0:
                                g2 -= 2 * old * np.log(old / e)
                        state[c] = old + d
                        if e > 0:
                            x2 += (old + d - e) ** 2 / e
                            if old + d > 0:
                                g2 += 2 * (old + d) * np.log((old + d) / e)
        if step >= burn_in:
            stats[step - burn_in] = (x2, g2)
    return stats, state


@_register_dataframe_method
def loglinear(
    data,
    x,
    interactions=None,
    count=None,
    structural_zeros=None,
    n_mcmc=10_000,
    burn_in=1_000,
    seed=None,
):
    """
    Hierarchical log-linear analysis of a multiway contingency table, with
    support for structural zeros and exact conditional tests.

    .. versionadded:: 0.8.0

    Parameters
    ----------
    data : :py:class:`pandas.DataFrame`
        The dataframe. By default each row is one observation; if ``count`` is
        given, each row is one cell of the contingency table and ``count``
        gives the cell counts.
    x : list of string
        The categorical variables (columns of ``data``) that define the
        dimensions of the contingency table. Must contain at least two
        columns. Rows with missing values on any of these columns are removed.
    interactions : list of tuple or None
        The interaction terms of the hierarchical log-linear model, e.g.
        ``[("channel", "ctype"), ("ctype", "plan")]``. The model is
        hierarchical: each interaction implies all of its lower-order subsets,
        and the main effects are always included. The sufficient margins of
        the model -- the margins that are preserved when fitting the expected
        frequencies and when sampling tables for the exact test -- are the
        margins defined by the (maximal) interaction terms. If None (default),
        the mutual independence model (main effects only) is fitted.
    count : string or None
        Name of a column of ``data`` with the cell counts, for data that are
        already aggregated into a contingency table (one row per cell
        combination). Counts must be non-negative integers. If None (default),
        each row of ``data`` counts as one observation.
    structural_zeros : :py:class:`pandas.DataFrame` or None
        The impossible cell combinations (structural zeros), given as a
        dataframe whose columns are a subset of ``x``. Every cell of the
        contingency table that matches a row of ``structural_zeros`` on all of
        its columns is treated as a structural zero: it is excluded from the
        fit (its expected frequency is exactly 0) and from the degrees of
        freedom, rather than being treated as an ordinary zero count. The
        observed count of a structural-zero cell must be zero.
    n_mcmc : int or None
        Number of Metropolis-Hastings samples used to estimate the exact
        conditional p-values (after ``burn_in`` discarded steps). Use 0 or
        None to skip the exact test (``p_exact`` is then NaN).
    burn_in : int
        Number of initial Metropolis-Hastings steps discarded as burn-in.
    seed : int or None
        Seed of the random number generator, for reproducible exact p-values.

    Returns
    -------
    expected : :py:class:`pandas.DataFrame`
        The fitted expected frequencies, in long format: one row per cell of
        the table, with the columns of ``x``, a ``'expected'`` column and a
        ``'structural_zero'`` flag.
    observed : :py:class:`pandas.DataFrame`
        The observed frequencies, in the same long format (``'observed'``
        column).
    stats : :py:class:`pandas.DataFrame`
        The goodness-of-fit tests, with one row per statistic (Pearson's
        chi-squared and the likelihood-ratio G-squared):

        * ``'test'``: The statistic name
        * ``'chi2'``: The test statistic
        * ``'dof'``: The degrees of freedom, adjusted for structural zeros
        * ``'p_asymp'``: The asymptotic chi-squared p-value
        * ``'p_exact'``: The exact conditional p-value (Monte Carlo estimate)
        * ``'mc_se'``: The Monte Carlo standard error of ``'p_exact'``

    See Also
    --------
    chi2_independence : Chi-squared independence tests between two variables.

    Notes
    -----
    **Hierarchical log-linear models.** The expected cell counts are
    :math:`\\log m_c = \\sum_{g \\in \\mathcal{G}} \\lambda_g(c)`, where
    :math:`\\mathcal{G}` is the set of interaction terms (generators). The
    model is fitted by iterative proportional fitting (IPF), which matches the
    expected margins to the observed margins for every generator; these
    margins are the minimal sufficient statistics of the model.

    **Structural zeros** are cells that are impossible by design (e.g. a
    pricing plan that is not offered through a channel). They are fundamentally
    different from *sampling* zeros (cells that simply got no observation):
    they are excluded from the fit, get an expected frequency of exactly 0,
    and reduce the degrees of freedom. The degrees of freedom are computed
    algebraically as the number of non-structural cells minus the rank of the
    model design matrix restricted to those cells, which automatically
    accounts for parameters that become non-estimable because of structural
    zeros.

    **Exact conditional test.** Conditioning on the sufficient margins removes
    the unknown model parameters: under the null hypothesis that the model
    holds, the conditional distribution of the table given its sufficient
    margins is proportional to :math:`1 / \\prod_c x_c!` (for Poisson,
    multinomial or product-multinomial sampling). The exact conditional
    p-value is the probability, under this distribution, of a goodness-of-fit
    statistic at least as extreme as the observed one. It is estimated here by
    Metropolis-Hastings sampling over the tables that share the observed
    sufficient margins and respect the structural zeros (Diaconis & Sturmfels
    1998), using the continuity-corrected estimator
    :math:`(1 + b) / (1 + n)` of Phipson & Smyth (2010), where :math:`b` is
    the number of sampled tables with a statistic at least as extreme. This is
    a genuine small-sample exact test, *not* the asymptotic chi-squared
    approximation; the reported Monte Carlo standard error quantifies the
    estimation error and can be reduced by increasing ``n_mcmc``. Note that
    the move set is guaranteed to connect all tables only for decomposable
    models and two-way tables; for general sparse multiway tables the chain
    might not explore the entire fiber, in which case the estimate can be
    biased towards the observed neighbourhood.

    .. warning :: The asymptotic p-value ``p_asymp`` is not reliable with
        small samples or sparse tables (a common guideline requires all
        expected frequencies to be at least 5). Use ``p_exact`` in that case.

    References
    ----------
    .. [1] Bishop, Y. M. M., Fienberg, S. E., & Holland, P. W. (1975).
           *Discrete multivariate analysis: theory and practice*. MIT Press.

    .. [2] Diaconis, P., & Sturmfels, B. (1998). Algebraic algorithms for
           sampling from conditional distributions. *The Annals of
           Statistics*, 26(1), 363-397.

    .. [3] Caffo, B. S. (2006). Exact hypothesis tests for log-linear models
           with exactLoglinTest. *R News*, 6(2), 10-13.

    .. [4] Phipson, B., & Smyth, G. K. (2010). Permutation p-values should
           never be zero: calculating exact p-values when permutations are
           randomly drawn. *Statistical Applications in Genetics and
           Molecular Biology*, 9(1).

    Examples
    --------
    A three-way table of sales channel x customer type x chosen plan, where
    the "trial" plan is not sold through the "partner" channel (a structural
    zero):

    >>> import pingouin as pg
    >>> import pandas as pd
    >>> cells = pd.DataFrame(
    ...     {
    ...         "channel": [
    ...             "web",
    ...             "web",
    ...             "web",
    ...             "web",
    ...             "web",
    ...             "web",
    ...             "partner",
    ...             "partner",
    ...             "partner",
    ...             "partner",
    ...         ],
    ...         "ctype": ["new", "new", "new", "old", "old", "old", "new", "new", "old", "old"],
    ...         "plan": [
    ...             "basic",
    ...             "pro",
    ...             "trial",
    ...             "basic",
    ...             "pro",
    ...             "trial",
    ...             "basic",
    ...             "pro",
    ...             "basic",
    ...             "pro",
    ...         ],
    ...         "n": [40, 25, 12, 30, 18, 4, 22, 15, 26, 13],
    ...     }
    ... )
    >>> sz = pd.DataFrame({"channel": ["partner"], "plan": ["trial"]})

    Fit the conditional independence model (plan independent of channel given
    customer type), which preserves the channel x ctype and ctype x plan
    margins:

    >>> expected, observed, stats = pg.loglinear(
    ...     cells,
    ...     x=["channel", "ctype", "plan"],
    ...     count="n",
    ...     interactions=[("channel", "ctype"), ("ctype", "plan")],
    ...     structural_zeros=sz,
    ...     seed=42,
    ... )
    >>> expected.round(2)
        channel ctype   plan  expected  structural_zero
    0       web   new  basic     39.51            False
    1       web   new    pro     25.49            False
    2       web   new  trial     12.00            False
    3       web   old  basic     30.90            False
    4       web   old    pro     17.10            False
    5       web   old  trial      4.00            False
    6   partner   new  basic     22.49            False
    7   partner   new    pro     14.51            False
    8   partner   new  trial      0.00             True
    9   partner   old  basic     25.10            False
    10  partner   old    pro     13.90            False
    11  partner   old  trial      0.00             True

    The structural zeros are excluded from the fit (expected count of exactly
    0) instead of being treated as ordinary zero counts, and the degrees of
    freedom are adjusted accordingly. The test summary reports both the
    asymptotic and the exact conditional p-values:

    >>> stats.round(4)
                 test    chi2  dof  p_asymp  p_exact  mc_se
    0         pearson  0.2056    2   0.9023   0.9388 0.0024
    1  log-likelihood  0.2059    2   0.9022   0.9388 0.0024

    With small samples or sparse tables, rely on ``p_exact`` rather than on
    the asymptotic ``p_asymp``.
    """
    # ----- Validation -----
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be a pandas DataFrame, but received {type(data).__name__}.")
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        raise ValueError("'x' must be a list of at least two column names of 'data'.")
    if len(set(x)) != len(x):
        raise ValueError("'x' contains duplicated column names.")
    if not all(isinstance(col, (str, int)) for col in x):
        raise TypeError("All elements of 'x' must be strings or ints.")
    if not all(col in data.columns for col in x):
        raise ValueError("columns of 'x' are not in dataframe.")
    if interactions is None:
        interactions = []
    if not isinstance(interactions, (list, tuple)):
        raise TypeError("'interactions' must be a list of tuples of column names.")
    if count is not None:
        if not isinstance(count, (str, int)):
            raise TypeError(
                f"'count' must be a string or int, but received {type(count).__name__}."
            )
        if count not in data.columns:
            raise ValueError("'count' is not a column of dataframe.")
    if structural_zeros is not None and not isinstance(structural_zeros, pd.DataFrame):
        raise TypeError("'structural_zeros' must be a pandas DataFrame or None.")
    if not (isinstance(burn_in, numbers.Integral) and burn_in >= 0):
        raise ValueError("'burn_in' must be a non-negative integer.")
    if n_mcmc is not None and not (isinstance(n_mcmc, numbers.Integral) and n_mcmc >= 0):
        raise ValueError("'n_mcmc' must be a non-negative integer or None.")

    # Rows with missing values on the table dimensions are removed
    data = data.dropna(subset=list(x))
    if count is not None:
        counts_col = data[count]
        if counts_col.isna().any():
            raise ValueError("Missing values are not allowed in the 'count' column.")
        if not np.issubdtype(counts_col.dtype, np.number):
            raise TypeError("The 'count' column must be numeric.")
        counts_np = counts_col.to_numpy(dtype=float)
        if (counts_np < 0).any() or not np.equal(np.mod(counts_np, 1), 0).all():
            raise ValueError("The 'count' column must contain non-negative integers.")
    else:
        counts_np = np.ones(data.shape[0])

    generators = _loglinear_generators(list(x), interactions)

    # ----- Contingency table on the full grid of factor levels -----
    levels = [pd.unique(data[col]) for col in x]
    if any(len(lv) == 0 for lv in levels):
        raise ValueError("No data; the contingency table is empty.")
    shape = tuple(len(lv) for lv in levels)
    n_cells = int(np.prod(shape))
    grid = pd.MultiIndex.from_product(levels, names=list(x))
    grid_df = grid.to_frame(index=False)
    # Flat cell index of every row of `data`
    codes = []
    for col, lv in zip(x, levels, strict=True):
        codes.append(data[col].map({lev: i for i, lev in enumerate(lv)}).to_numpy())
    cell_of_row = np.ravel_multi_index(codes, shape)
    obs = np.bincount(cell_of_row, weights=counts_np, minlength=n_cells)
    if obs.sum() == 0:
        raise ValueError("No data; the total count of the contingency table is zero.")

    # ----- Structural zeros -----
    structural = np.zeros(n_cells, dtype=bool)
    if structural_zeros is not None and structural_zeros.shape[0] > 0:
        if structural_zeros.shape[1] == 0 or not set(structural_zeros.columns) <= set(x):
            raise ValueError("The columns of 'structural_zeros' must be a subset of 'x'.")
        for _, row in structural_zeros.iterrows():
            match = np.ones(n_cells, dtype=bool)
            for col in structural_zeros.columns:
                match &= (grid_df[col] == row[col]).to_numpy()
            if not match.any():
                warnings.warn(
                    f"A row of 'structural_zeros' matches no cell of the table: {dict(row)}",
                    stacklevel=2,
                )
            structural |= match
    if structural.all():
        raise ValueError("All cells of the table are structural zeros.")
    bad = structural & (obs > 0)
    if bad.any():
        raise ValueError(
            f"Structural-zero cells must have an observed count of zero, but "
            f"{int(bad.sum())} structural-zero cell(s) have positive counts."
        )
    allowed = ~structural

    # ----- Margins (buckets) of each generator -----
    grid_codes = {
        col: grid_df[col].map({lev: i for i, lev in enumerate(lv)}).to_numpy()
        for col, lv in zip(x, levels, strict=True)
    }
    gen_buckets, gen_sizes = [], []
    for gen in generators:
        cols = sorted(gen, key=str)
        sizes = [len(levels[list(x).index(c)]) for c in cols]
        gen_buckets.append(np.ravel_multi_index([grid_codes[c] for c in cols], sizes))
        gen_sizes.append(int(np.prod(sizes)))

    # ----- Fit the expected frequencies (IPF, structural zeros excluded) -----
    expected = _loglinear_ipf(obs, allowed, gen_buckets, gen_sizes)

    # ----- Degrees of freedom: non-structural cells minus design matrix rank -----
    design = np.zeros((int(allowed.sum()), sum(gen_sizes)))
    row_ok = np.flatnonzero(allowed)
    col_start = 0
    for bucket, size in zip(gen_buckets, gen_sizes, strict=True):
        design[np.arange(row_ok.size), col_start + bucket[row_ok]] = 1.0
        col_start += size
    dof = int(allowed.sum() - np.linalg.matrix_rank(design))
    if dof <= 0:
        warnings.warn(
            "The model is saturated (or over-parameterized given the structural "
            "zeros): the degrees of freedom are not positive and the "
            "goodness-of-fit tests are not informative.",
            stacklevel=2,
        )

    # ----- Goodness-of-fit statistics -----
    pos = expected > 0
    if np.any(obs[~pos] > 0):
        raise ValueError(
            "The model assigns an expected frequency of zero to cells with a "
            "positive observed count; the model cannot fit these data."
        )
    x2_obs = np.sum((obs[pos] - expected[pos]) ** 2 / expected[pos])
    g2_obs = 2 * np.sum(
        obs[pos & (obs > 0)] * np.log(obs[pos & (obs > 0)] / expected[pos & (obs > 0)])
    )
    if dof > 0:
        p_asymp = (sp_chi2.sf(x2_obs, dof), sp_chi2.sf(g2_obs, dof))
    else:
        p_asymp = (np.nan, np.nan)

    # ----- Exact conditional test (Metropolis-Hastings over the fiber) -----
    p_exact = (np.nan, np.nan)
    mc_se = (np.nan, np.nan)
    if n_mcmc and dof > 0:
        rng = np.random.default_rng(seed)
        gen_sets = [frozenset(list(x).index(c) for c in gen) for gen in generators]
        samples, _ = _loglinear_mcmc(
            obs, expected, allowed, shape, gen_sets, gen_buckets, int(n_mcmc), int(burn_in), rng
        )
        if samples is None:
            warnings.warn(
                "No margin-preserving move is available for this model; the "
                "exact conditional test cannot be computed.",
                stacklevel=2,
            )
        else:
            est, se = [], []
            for stat_obs, stat_seq in zip((x2_obs, g2_obs), samples.T, strict=True):
                p = (1 + np.sum(stat_seq >= stat_obs - 1e-10)) / (1 + len(stat_seq))
                est.append(p)
                se.append(np.sqrt(p * (1 - p) / len(stat_seq)))
            p_exact, mc_se = tuple(est), tuple(se)

    # ----- Outputs -----
    expected_df = grid_df.copy()
    expected_df["expected"] = expected
    expected_df["structural_zero"] = structural
    observed_df = grid_df.copy()
    observed_df["observed"] = obs.astype(np.int64)
    observed_df["structural_zero"] = structural
    stats = pd.DataFrame(
        {
            "test": ["pearson", "log-likelihood"],
            "chi2": [x2_obs, g2_obs],
            "dof": dof,
            "p_asymp": list(p_asymp),
            "p_exact": list(p_exact),
            "mc_se": list(mc_se),
        }
    )
    return expected_df, observed_df, _postprocess_dataframe(stats)
