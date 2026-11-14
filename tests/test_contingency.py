from math import factorial

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2 as sp_chi2
from scipy.stats import chi2_contingency

import pingouin as pg

df_ind = pg.read_dataset("chi2_independence")
df_mcnemar = pg.read_dataset("chi2_mcnemar")

data_ct = pd.DataFrame(
    {
        "A": [0, 1, 0, 0],
        "B": [False, True, False, False],
        "C": [1, 2, 3, 4],
        "D": ["No", "Yes", "No", "No"],
        "E": ["y", "y", "y", "y"],
    }
)


@pytest.mark.filterwarnings("ignore:Low count:UserWarning")
def test_chi2_independence():
    """Test function chi2_independence."""
    # Setup
    rng = np.random.RandomState(42)
    mean, cov = [0.5, 0.5], [(1, 0.6), (0.6, 1)]
    x, y = rng.multivariate_normal(mean, cov, 30).T
    data = pd.DataFrame({"x": x, "y": y})
    mask_class_1 = data > 0.5
    data[mask_class_1] = 1
    data[~mask_class_1] = 0

    # Comparing results with SciPy
    with pytest.warns(UserWarning, match="Low count"):
        _, _, stats = pg.chi2_independence(data, x="x", y="y")
    contingency_table = pd.crosstab(data["x"], data["y"])
    for i in stats.index:
        lambda_ = stats.at[i, "lambda"]
        dof = stats.at[i, "dof"]
        chi2 = stats.at[i, "chi2"]
        p = round(stats.at[i, "pval"], 6)
        sp_chi2, sp_p, sp_dof, _ = chi2_contingency(contingency_table, lambda_=lambda_)
        np.testing.assert_allclose([chi2, p, dof], [sp_chi2, sp_p, sp_dof], rtol=1e-4)

    # Yates' correction cannot exceed the deviation between observed and expected
    data_small = pd.DataFrame(
        {"x": [0] * 20 + [1] * 21, "y": [0] * 10 + [1] * 10 + [0] * 10 + [1] * 11}
    )
    _, _, stats = pg.chi2_independence(data_small, x="x", y="y")
    sp_chi2, sp_p, _, _ = chi2_contingency(pd.crosstab(data_small["x"], data_small["y"]))
    np.testing.assert_allclose(stats.loc[0, ["chi2", "pval"]].astype(float), [sp_chi2, sp_p])

    # Testing resilience to NaN
    _, _, stats_full = pg.chi2_independence(data, x="x", y="y")
    data_nan = pd.concat(
        [data, pd.DataFrame({"x": [np.nan] * 20, "y": [1.0] * 20})], ignore_index=True
    )
    _, _, stats_nan = pg.chi2_independence(data_nan, x="x", y="y")
    # Rows with missing values are not counted in the sample size (Cramer's V and power)
    pd.testing.assert_frame_equal(stats_full, stats_nan)
    mask_nan = rng.random_sample(data.shape) > 0.8  # ~20% NaN values
    data[mask_nan] = np.nan
    _, observed, stats_nan = pg.chi2_independence(data, x="x", y="y")
    assert observed.to_numpy().sum() == data.dropna().shape[0]
    pd.testing.assert_frame_equal(stats_nan, pg.chi2_independence(data.dropna(), "x", "y")[2])

    # Testing validations
    def expect_error(error, *params, **kwargs):
        with pytest.raises(error):
            pg.chi2_independence(*params, **kwargs)

    expect_error(TypeError, 1, "x", "y")  # Not a pd.DataFrame
    expect_error(TypeError, data, x, "y")  # Not a string
    expect_error(TypeError, data, "x", y)  # Not a string
    expect_error(ValueError, data, "x", "z")  # Not a column of data
    expect_error(TypeError, data, "x", "y", correction=1)  # Not a boolean

    # Testing "no data" ValueError
    data["x"] = np.nan
    with pytest.raises(ValueError):
        pg.chi2_independence(data, x="x", y="y")

    # Testing degenerated case (observed == expected)
    data["x"] = 1
    data["y"] = 1
    expected, observed, stats = pg.chi2_independence(data, "x", "y")
    assert expected.iloc[0, 0] == observed.iloc[0, 0]
    assert stats.at[0, "dof"] == 0
    for i in stats.index:
        chi2 = stats.at[i, "chi2"]
        p = stats.at[i, "pval"]
        assert (chi2, p) == (0.0, 1.0)

    # Testing warning on low count
    data.iloc[0, 0] = 0
    with pytest.warns(UserWarning):
        pg.chi2_independence(data, "x", "y")

    # Comparing results with R
    # 2 x 2 contingency table (dof = 1)
    # >>> tbl = table(df$sex, df$target)
    # >>> chisq.test(tbl, correct = TRUE)
    # >>> cramersV(tbl)
    _, _, stats = pg.chi2_independence(df_ind, "sex", "target")
    assert round(stats.at[0, "chi2"], 3) == 22.717
    assert stats.at[0, "dof"] == 1
    assert np.isclose(stats.at[0, "pval"], 1.877e-06)
    assert round(stats.at[0, "cramer"], 2) == 0.27

    # 4 x 2 contingency table
    _, _, stats = pg.chi2_independence(df_ind, "cp", "target")
    assert round(stats.at[0, "chi2"], 3) == 81.686
    assert stats.at[0, "dof"] == 3.0
    assert stats.at[0, "pval"] < 2.2e-16
    assert round(stats.at[0, "cramer"], 3) == 0.519
    assert np.isclose(stats.at[0, "power"], 1.0)


def test_chi2_mcnemar():
    """Test function chi2_mcnemar."""
    # Setup
    rng = np.random.RandomState(42)
    mean, cov = [0.5, 0.5], [(1, 0.6), (0.6, 1)]
    x, y = rng.multivariate_normal(mean, cov, 30).T
    data = pd.DataFrame({"x": x, "y": y})
    mask_class_1 = data > 0.5
    data[mask_class_1] = 1
    data[~mask_class_1] = 0

    # Testing validations
    def expect_error(error, *params):
        with pytest.raises(error):
            pg.chi2_mcnemar(*params)

    expect_error(TypeError, 1, "x", "y")  # Not a pd.DataFrame
    expect_error(TypeError, data, x, "y")  # Not a string
    expect_error(TypeError, data, "x", y)  # Not a string
    expect_error(ValueError, data, "x", "z")  # Not a column of data

    # Testing happy-day
    observed, stats = pg.chi2_mcnemar(data, "x", "y")
    assert observed.to_numpy().sum() == data.shape[0]
    np.testing.assert_array_equal(observed.to_numpy(), pd.crosstab(data["x"], data["y"]))
    assert stats.columns.tolist() == ["chi2", "dof", "p_approx", "p_exact"]
    assert 0 <= stats.at["mcnemar", "p_approx"] <= 1

    # Testing NaN incompatibility
    data.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        pg.chi2_mcnemar(data, "x", "y")

    # Testing invalid dichotomous value
    data.iloc[0, 0] = 3
    with pytest.raises(ValueError):
        pg.chi2_mcnemar(data, "x", "y")

    # Testing error when b == 0 and c == 0
    data = pd.DataFrame({"x": [0, 0, 0, 1, 1, 1], "y": [0, 0, 0, 1, 1, 1]})
    with pytest.raises(ValueError):
        pg.chi2_mcnemar(data, "x", "y")

    # All the subjects switched: the crosstab has a single non-zero (discordant) cell
    observed, stats = pg.chi2_mcnemar(pd.DataFrame({"x": [0] * 10, "y": [1] * 10}), "x", "y")
    np.testing.assert_array_equal(observed.to_numpy(), [[0, 10], [0, 0]])
    assert np.isclose(stats.at["mcnemar", "p_exact"], 2 * 0.5**10)
    # NumPy scalars in an object column
    data = pd.DataFrame({"x": np.array([0, 1] * 5, dtype=object), "y": [np.int64(1)] * 10})
    observed, stats = pg.chi2_mcnemar(data, "x", "y")
    observed_int, stats_int = pg.chi2_mcnemar(data.astype(int), "x", "y")
    assert observed.equals(observed_int)
    assert stats.equals(stats_int)

    # Comparing results with R
    # 2 x 2 contingency table (dof = 1)
    # >>> tbl = table(df$treatment_X, df$treatment_Y)
    # >>> mcnemar.test(tbl, correct = TRUE)
    _, stats = pg.chi2_mcnemar(df_mcnemar, "treatment_X", "treatment_Y")
    assert round(stats.at["mcnemar", "chi2"], 3) == 20.021
    assert stats.at["mcnemar", "dof"] == 1
    assert np.isclose(stats.at["mcnemar", "p_approx"], 7.66e-06)
    # Results are compared to the exact2x2 R package
    # >>> exact2x2(tbl, paired = TRUE, midp = FALSE)
    assert np.isclose(stats.at["mcnemar", "p_exact"], 3.305e-06)
    # midp gives slightly different results
    # assert np.allclose(stats.at['mcnemar', 'p_mid'], 3.305e-06)


def test_dichotomize_series():
    """Test function _dichotomize_series."""
    # Integer
    a = pg.contingency._dichotomize_series(data_ct, "A").to_numpy()
    b = pg.contingency._dichotomize_series(data_ct, "B").to_numpy()
    d = pg.contingency._dichotomize_series(data_ct, "D").to_numpy()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(b, d)
    with pytest.raises(ValueError):
        pg.contingency._dichotomize_series(data_ct, "C")
    # Strings that are not recognized as dichotomous values
    with pytest.raises(ValueError, match="maybe"):
        pg.contingency._dichotomize_series(pd.DataFrame({"x": ["yes", "maybe"]}), "x")


def test_dichotomous_crosstab():
    """Test function dichotomous_crosstab."""
    # Integer
    d1 = pg.dichotomous_crosstab(data_ct, "A", "B")
    d2 = pg.dichotomous_crosstab(data_ct, "A", "D")
    assert d1.equals(d2)
    np.testing.assert_array_equal(
        pg.dichotomous_crosstab(data_ct, "E", "A").to_numpy(),
        pg.dichotomous_crosstab(data_ct, "A", "E").to_numpy().T,
    )
    # Missing levels are filled with zero counts
    np.testing.assert_array_equal(
        pg.dichotomous_crosstab(data_ct, "E", "E").to_numpy(), [[0, 0], [0, 4]]
    )
    np.testing.assert_array_equal(
        pg.dichotomous_crosstab(data_ct, "A", "E").to_numpy(), [[0, 3], [0, 1]]
    )


# 3-way count table (channel x customer type x plan); the "trial" plan is not
# sold through the "partner" channel (structural zeros).
cells_3way = pd.DataFrame(
    {
        "channel": ["web"] * 6 + ["partner"] * 4,
        "ctype": ["new"] * 3 + ["old"] * 3 + ["new"] * 2 + ["old"] * 2,
        "plan": ["basic", "pro", "trial"] * 2 + ["basic", "pro"] * 2,
        "n": [40, 25, 12, 30, 18, 4, 22, 15, 26, 13],
    }
)
sz_3way = pd.DataFrame({"channel": ["partner"], "plan": ["trial"]})


@pytest.mark.filterwarnings("ignore:Low count:UserWarning")
def test_loglinear():
    """Test function loglinear."""
    # ----- Two-way independence: equivalence with chi2_independence -----
    expected, observed, stats = pg.loglinear(df_ind, x=["sex", "target"], n_mcmc=None, seed=42)
    exp_chi2, _, stats_chi2 = pg.chi2_independence(df_ind, "sex", "target", correction=False)
    exp_wide = expected.pivot(index="sex", columns="target", values="expected")
    exp_wide = exp_wide.sort_index().sort_index(axis=1)
    np.testing.assert_allclose(
        exp_wide.to_numpy(), exp_chi2.sort_index().sort_index(axis=1).to_numpy(), rtol=1e-8
    )
    assert stats.at[0, "test"] == "pearson"
    assert stats.at[1, "test"] == "log-likelihood"
    # Pearson X2 and likelihood-ratio G2 match chi2_independence (lambda 1 and 0)
    assert np.isclose(stats.at[0, "chi2"], stats_chi2.at[0, "chi2"])
    assert np.isclose(stats.at[1, "chi2"], stats_chi2.at[2, "chi2"])
    assert stats.at[0, "dof"] == stats_chi2.at[0, "dof"] == 1
    assert np.isclose(stats.at[0, "p_asymp"], stats_chi2.at[0, "pval"])
    assert np.isclose(stats.at[1, "p_asymp"], stats_chi2.at[2, "pval"])
    # No exact test requested
    assert stats["p_exact"].isna().all() and stats["mc_se"].isna().all()
    # Observed table matches the crosstab, no structural zeros
    obs_wide = observed.pivot(index="sex", columns="target", values="observed")
    obs_wide = obs_wide.sort_index().sort_index(axis=1)
    np.testing.assert_array_equal(obs_wide.to_numpy(), pd.crosstab(df_ind.sex, df_ind.target))
    assert not observed["structural_zero"].any()

    # ----- Three-way model with interactions: sufficient margins are preserved -----
    expected, observed, stats = pg.loglinear(
        cells_3way,
        x=["channel", "ctype", "plan"],
        count="n",
        interactions=[("channel", "ctype"), ("ctype", "plan")],
        structural_zeros=sz_3way,
        seed=42,
    )
    for margins in (["channel", "ctype"], ["ctype", "plan"], ["channel"], ["ctype"], ["plan"]):
        exp_margin = expected.groupby(margins, observed=True)["expected"].sum()
        obs_margin = observed.groupby(margins, observed=True)["observed"].sum()
        np.testing.assert_allclose(exp_margin.to_numpy(), obs_margin.to_numpy(), atol=1e-6)
    # Structural zeros are excluded from the fit, not treated as zero counts
    sz_cells = expected[expected["structural_zero"]]
    assert len(sz_cells) == 2
    assert (sz_cells["expected"] == 0).all()
    assert (observed.loc[observed["structural_zero"], "observed"] == 0).all()
    # dof: 10 non-structural cells - rank 8 (4 + 6 - 2 for the shared ctype margin)
    assert stats.at[0, "dof"] == 2
    assert 0 < stats.at[0, "p_exact"] <= 1 and stats.at[0, "mc_se"] > 0

    # Declaring the structural zeros changes the fit (they are not sampling zeros)
    expected_nosz, _, stats_nosz = pg.loglinear(
        cells_3way,
        x=["channel", "ctype", "plan"],
        count="n",
        interactions=[("channel", "ctype"), ("ctype", "plan")],
        n_mcmc=None,
    )
    assert stats_nosz.at[0, "dof"] == 4  # 12 cells - rank 8
    assert not np.allclose(expected["expected"], expected_nosz["expected"])

    # ----- Cross-validation against a Poisson GLM (statsmodels) -----
    # The expected frequencies, dof, X2 and G2 must match the equivalent GLM
    # fitted on the non-structural cells only.
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    fit = expected.loc[~expected["structural_zero"], ["channel", "ctype", "plan", "expected"]]
    fit["n"] = observed.loc[~observed["structural_zero"], "observed"].to_numpy()
    glm = smf.glm("n ~ channel*ctype + ctype*plan", data=fit, family=sm.families.Poisson()).fit()
    np.testing.assert_allclose(fit["expected"], glm.fittedvalues, atol=1e-6)
    assert glm.df_resid == stats.at[0, "dof"]
    assert np.isclose(glm.pearson_chi2, stats.at[0, "chi2"])
    assert np.isclose(glm.deviance, stats.at[1, "chi2"])

    # ----- Structural zeros reduce the degrees of freedom (quasi-independence) -----
    # 3x3 table with structural zeros on the diagonal: dof = 6 cells - rank 5 = 1
    tab_qi = pd.DataFrame(
        {
            "r": [0, 0, 1, 1, 2, 2],
            "c": [1, 2, 0, 2, 0, 1],
            "n": [5, 3, 2, 6, 4, 1],
        }
    )
    sz_diag = pd.DataFrame({"r": [0, 1, 2], "c": [0, 1, 2]})
    expected, _, stats = pg.loglinear(
        tab_qi, x=["r", "c"], count="n", structural_zeros=sz_diag, n_mcmc=None
    )
    assert stats.at[0, "dof"] == 1
    assert (expected.loc[expected["structural_zero"], "expected"] == 0).all()
    # Row and column margins are preserved on the non-structural cells
    for margins in (["r"], ["c"]):
        np.testing.assert_allclose(
            expected.groupby(margins)["expected"].sum().to_numpy(),
            tab_qi.groupby(margins)["n"].sum().to_numpy(),
            atol=1e-6,
        )
    # p_asymp matches the chi-squared survival function with the adjusted dof
    assert np.isclose(stats.at[0, "p_asymp"], sp_chi2.sf(stats.at[0, "chi2"], 1))

    # ----- Count-column input equals long-format input -----
    rows = cells_3way.loc[cells_3way.index.repeat(cells_3way["n"]), ["channel", "ctype", "plan"]]
    kwargs = dict(
        x=["channel", "ctype", "plan"],
        interactions=[("channel", "ctype"), ("ctype", "plan")],
        structural_zeros=sz_3way,
        seed=42,
    )
    exp_count, _, stats_count = pg.loglinear(cells_3way, count="n", **kwargs)
    exp_long, _, stats_long = pg.loglinear(rows, **kwargs)
    pd.testing.assert_frame_equal(exp_count, exp_long)
    pd.testing.assert_frame_equal(stats_count, stats_long)

    # ----- Hierarchical reduction: redundant terms do not change the model -----
    exp1, _, stats1 = pg.loglinear(
        cells_3way,
        x=["channel", "ctype", "plan"],
        count="n",
        interactions=[("channel", "ctype"), ("channel",), ("ctype",)],
        structural_zeros=sz_3way,
        n_mcmc=None,
    )
    exp2, _, stats2 = pg.loglinear(
        cells_3way,
        x=["channel", "ctype", "plan"],
        count="n",
        interactions=[("channel", "ctype")],
        structural_zeros=sz_3way,
        n_mcmc=None,
    )
    pd.testing.assert_frame_equal(exp1, exp2)
    pd.testing.assert_frame_equal(stats1, stats2)

    # ----- Rows with missing values are removed -----
    cells_nan = pd.concat(
        [
            cells_3way,
            pd.DataFrame({"channel": [None], "ctype": ["new"], "plan": ["pro"], "n": [5]}),
        ],
        ignore_index=True,
    )
    exp_nan, _, _ = pg.loglinear(
        cells_nan,
        x=["channel", "ctype", "plan"],
        count="n",
        interactions=[("channel", "ctype"), ("ctype", "plan")],
        structural_zeros=sz_3way,
        n_mcmc=None,
    )
    pd.testing.assert_frame_equal(exp_nan, exp_count)

    # ----- Saturated model: dof = 0, tests are not informative -----
    with pytest.warns(UserWarning, match="saturated"):
        _, _, stats_sat = pg.loglinear(
            df_ind, x=["sex", "target"], interactions=[("sex", "target")]
        )
    assert stats_sat.at[0, "dof"] == 0
    assert stats_sat["p_asymp"].isna().all() and stats_sat["p_exact"].isna().all()

    # ----- Also registered as a DataFrame method -----
    _, _, stats_method = df_ind.loglinear(x=["sex", "target"], n_mcmc=None)
    assert stats_method.at[0, "dof"] == 1


def test_loglinear_exact_conditional():
    """Test the exact conditional p-value of loglinear against enumeration."""
    # 2x2 table with fixed margins: the fiber has only 6 tables, enumerable by hand
    tab = pd.DataFrame({"r": [0, 0, 1, 1], "c": [0, 1, 0, 1], "n": [2, 5, 4, 1]})
    expected, _, stats = pg.loglinear(tab, x=["r", "c"], count="n", n_mcmc=20000, seed=42)
    exp2x2 = expected["expected"].to_numpy().reshape(2, 2)

    def exact_pvalue(stat_obs, stat_fun):
        """Exact conditional p-value by enumeration of the 2x2 fiber."""
        weights, stats_fiber = [], []
        for a in range(1, 7):  # tables [[a, 7-a], [6-a, a-1]] have the same margins
            table = np.array([[a, 7 - a], [6 - a, a - 1]], dtype=float)
            weights.append(np.prod([1 / factorial(int(v)) for v in table.ravel()]))
            stats_fiber.append(stat_fun(table))
        weights = np.asarray(weights)
        weights /= weights.sum()
        return weights[np.asarray(stats_fiber) >= stat_obs - 1e-10].sum()

    def pearson(table):
        return ((table - exp2x2) ** 2 / exp2x2).sum()

    def loglik(table):
        pos = table > 0
        return 2 * (table[pos] * np.log(table[pos] / exp2x2[pos])).sum()

    p_true_x2 = exact_pvalue(stats.at[0, "chi2"], pearson)
    p_true_g2 = exact_pvalue(stats.at[1, "chi2"], loglik)
    assert np.isclose(p_true_x2, 32 / 132)  # hand-checked value
    # The Monte Carlo estimates match the enumerated exact p-values
    assert abs(stats.at[0, "p_exact"] - p_true_x2) < 4 * stats.at[0, "mc_se"] + 1e-3
    assert abs(stats.at[1, "p_exact"] - p_true_g2) < 4 * stats.at[1, "mc_se"] + 1e-3
    # ... while the asymptotic p-values are far off on this small sample
    assert abs(stats.at[0, "p_asymp"] - p_true_x2) > 0.1

    # Reproducibility: same seed, same p-values
    _, _, stats_rep = pg.loglinear(tab, x=["r", "c"], count="n", n_mcmc=20000, seed=42)
    pd.testing.assert_frame_equal(stats, stats_rep)

    # The MCMC chain preserves the sufficient margins and the structural zeros
    # 2x3 table with a structural zero at cell (0, 1), independence model
    obs = np.array([2, 0, 4, 1, 3, 2], dtype=float)
    allowed = np.array([True, False, True, True, True, True])
    gen_sets = [frozenset({0}), frozenset({1})]
    gen_buckets = [np.repeat(np.arange(2), 3), np.tile(np.arange(3), 2)]
    rng = np.random.default_rng(12345)
    _, last = pg.contingency._loglinear_mcmc(
        obs, obs.copy(), allowed, (2, 3), gen_sets, gen_buckets, 2000, 200, rng
    )
    assert last[1] == 0  # structural zero untouched
    tab_last = last.reshape(2, 3)
    np.testing.assert_array_equal(tab_last.sum(axis=0), obs.reshape(2, 3).sum(axis=0))
    np.testing.assert_array_equal(tab_last.sum(axis=1), obs.reshape(2, 3).sum(axis=1))
    assert not np.array_equal(last, obs)  # the chain did move


def test_loglinear_validation():
    """Test the input validation of loglinear."""
    kwargs = dict(x=["channel", "ctype", "plan"], count="n", n_mcmc=None)

    def expect_error(error, *params, **kw):
        with pytest.raises(error):
            pg.loglinear(*params, **kw)

    expect_error(TypeError, 1, **kwargs)  # data not a DataFrame
    expect_error(ValueError, cells_3way, x=["channel"], count="n")  # less than two columns
    expect_error(ValueError, cells_3way, x=["channel", "channel"], count="n")  # duplicated
    expect_error(TypeError, cells_3way, x=["channel", 1.5, "plan"], count="n")  # invalid name
    expect_error(ValueError, cells_3way, x=["channel", "ctype", "unknown"], count="n")
    expect_error(TypeError, cells_3way, interactions="channel", **kwargs)  # not a list
    expect_error(TypeError, cells_3way, interactions=["channel"], **kwargs)  # string term
    expect_error(ValueError, cells_3way, interactions=[()], **kwargs)  # empty term
    expect_error(ValueError, cells_3way, interactions=[("channel", "unknown")], **kwargs)
    expect_error(TypeError, cells_3way, **{**kwargs, "count": 1.5})  # count not a name
    expect_error(ValueError, cells_3way, **{**kwargs, "count": "unknown"})
    expect_error(TypeError, cells_3way, structural_zeros=[("partner", "trial")], **kwargs)
    expect_error(ValueError, cells_3way, structural_zeros=pd.DataFrame({"unknown": [1]}), **kwargs)
    expect_error(ValueError, cells_3way, **{**kwargs, "burn_in": -1})
    expect_error(ValueError, cells_3way, **{**kwargs, "n_mcmc": -5})

    # Invalid counts
    expect_error(ValueError, cells_3way.assign(n=-1), **kwargs)  # negative
    expect_error(ValueError, cells_3way.assign(n=1.5), **kwargs)  # non-integer
    expect_error(ValueError, cells_3way.assign(n=np.nan), **kwargs)  # missing
    expect_error(TypeError, cells_3way.assign(n="a"), **kwargs)  # non-numeric
    expect_error(ValueError, cells_3way.assign(n=0), **kwargs)  # all-zero table

    # Structural-zero cells must have a zero observed count
    sz_bad = pd.DataFrame({"channel": ["web"], "plan": ["basic"]})
    expect_error(ValueError, cells_3way, structural_zeros=sz_bad, **kwargs)
    # All cells cannot be structural zeros
    sz_all = cells_3way[["channel", "ctype", "plan"]].drop_duplicates()
    sz_all = pd.concat(
        [
            sz_all,
            pd.DataFrame(
                {"channel": ["partner"] * 2, "ctype": ["new", "old"], "plan": ["trial"] * 2}
            ),
        ]
    )
    expect_error(ValueError, cells_3way, structural_zeros=sz_all, **kwargs)
    # A structural-zero row that matches no cell warns
    with pytest.warns(UserWarning, match="matches no cell"):
        pg.loglinear(
            cells_3way,
            structural_zeros=pd.DataFrame({"channel": ["pigeon"]}),
            **kwargs,
        )
    # No data at all
    expect_error(ValueError, cells_3way.assign(channel=None), **kwargs)
