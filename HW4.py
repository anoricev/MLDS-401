# -*- coding: utf-8 -*-
"""
MLDS 401 - Homework 4
"""


import shlex
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas.plotting import scatter_matrix


# Q1
# ------------------------------------------------------------
# 1) Robust loader for Auto.txt that preserves the quoted 'name'
# ------------------------------------------------------------
def load_auto_txt(path: str = "Auto.txt") -> pd.DataFrame:
    """
    Parse Auto.txt which is whitespace-delimited with a quoted 'name' field.
    We use shlex to respect quotes so car names with spaces remain one field.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Header: split on whitespace (no quotes in header)
    cols = shlex.split(lines[0])
    # Expect 9 columns:
    # mpg cylinders displacement horsepower weight acceleration year origin name
    expected = 9
    for ln in lines[1:]:
        parts = shlex.split(ln)  # keeps the quoted car name together
        if len(parts) < expected:
            # Skip malformed short lines
            continue
        # Take only the first 9 tokens in case there are trailing artifacts
        parts = parts[:expected]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=cols)

    # Coerce numeric columns; keep 'name' as string
    num_cols = [
        'mpg', 'cylinders', 'displacement', 'horsepower',
        'weight', 'acceleration', 'year', 'origin'
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows that failed numeric coercion
    df = df.dropna(subset=num_cols).copy()

    # Standardize types for some integer-like fields
    df['cylinders'] = df['cylinders'].astype(int)
    df['year']      = df['year'].astype(int)
    df['origin']    = df['origin'].astype(int)

    return df


# ------------------------------------------------------------
# 2) Load data and create transforms
# ------------------------------------------------------------
auto = load_auto_txt("Auto.txt")

# Create transformed variables required by Q1
auto['logmpg']    = np.log(auto['mpg'])
auto['logdisp']   = np.log(auto['displacement'])
auto['logweight'] = np.log(auto['weight'])

# (Optional) quick sanity check
#print(auto.head())


# ------------------------------------------------------------
# 3) Q1(a): OLS regression
#     log(mpg) ~ cylinders + log(displacement) + log(weight) + year
# ------------------------------------------------------------
fit_full = smf.ols('logmpg ~ cylinders + logdisp + logweight + year', data=auto).fit()
print("\n=== Q1(a) OLS: log(mpg) ~ cylinders + logdisp + logweight + year ===")
print(fit_full.summary())


# ------------------------------------------------------------
# Scatterplot matrix (not explicitly required in Q1 but helpful for overview)
# You can comment this out if not needed in your final PDF.
# ------------------------------------------------------------
plt.figure(figsize=(9, 9))
scatter_matrix(
    auto[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'year']].dropna(),
    figsize=(9, 9),
    diagonal='kde'
)
plt.suptitle("Scatterplot Matrix (overview)", y=1.02)
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 4) Q1(b): Diagnostic plots for the fitted model
# ------------------------------------------------------------
resid  = fit_full.resid
fitted = fit_full.fittedvalues

# Residuals vs Fitted
plt.figure()
plt.scatter(fitted, resid, alpha=0.6)
plt.axhline(0, color='gray', lw=1)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Q1(b) Residuals vs Fitted')
plt.tight_layout()
plt.show()

# Scale-Location plot (rough heteroskedasticity check)
plt.figure()
plt.scatter(fitted, np.sqrt(np.abs(resid)), alpha=0.6)
plt.xlabel('Fitted values')
plt.ylabel('sqrt(|Residual|)')
plt.title('Q1(b) Scale-Location')
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 5) Q1(c): Variance Inflation Factors (VIF)
#    We compute VIF on the design matrix used by statsmodels.
#    VIF=1: no collinearity; VIF>5~10: potentially concerning.
# ------------------------------------------------------------
X = fit_full.model.exog
vifs = [(name, variance_inflation_factor(X, i))
        for i, name in enumerate(fit_full.model.exog_names)]

print("\n=== Q1(c) VIF (including intercept) ===")
for n, v in vifs:
    print(f"{n:>12s}: {v:8.3f}")


# ------------------------------------------------------------
# 6) Q1(d): Drop 'weight' and refit
#    Compare R^2 and coefficients to the full model
# ------------------------------------------------------------
fit_drop_w = smf.ols('logmpg ~ cylinders + logdisp + year', data=auto).fit()

print("\n=== Q1(d) Drop 'weight' and refit ===")
print("R^2 (full model)     :", round(fit_full.rsquared, 4))
print("R^2 (drop weight)    :", round(fit_drop_w.rsquared, 4))
print("\nSummary (drop weight):")
print(fit_drop_w.summary())


# ------------------------------------------------------------
# 7) Notes for write-up (no code required for part (e))
# ------------------------------------------------------------
"""
Q1(e) causal-discussion notes (to include in your PDF write-up, not code):
- A plausible DAG:
    cylinders -> displacement -> horsepower -> mpg
    cylinders -> weight -> mpg
    (vehicle class/size) -> cylinders, displacement, weight, horsepower, mpg
- If the target is the TOTAL effect of cylinders on mpg, avoid controlling mediators
  (weight/displacement/horsepower). Control only true confounders (common causes).
"""


# Q2
# ---------- 1) Generate data ----------
np.random.seed(0)
n = 500
w = np.random.uniform(0, 5, n)
x = w + np.random.normal(0, 1, n)
y = 4 + 2*x - 3*w + np.random.normal(0, 1, n)

df2 = pd.DataFrame({'x': x, 'w': w, 'y': y})

# ---------- 2) (b) Correlation and basic descriptive stats ----------
print("=== Q2 (b) Correlation matrix ===")
print(df2.corr().round(3), "\n")

print("=== Q2 (b) Descriptive stats (min, max, mean, std) ===")
print(df2.describe().loc[['min', 'max', 'mean', 'std']].round(3), "\n")

# ---------- 3) (c) OLS: y ~ x ----------
fit_c = smf.ols('y ~ x', data=df2).fit()
print("=== Q2 (c) OLS: y ~ x ===")
print(fit_c.summary())
print("95% CI for slope of x:", fit_c.conf_int().loc['x'].round(3).tolist(), "\n")

# ---------- 4) (d) OLS: y ~ x + w ----------
fit_d = smf.ols('y ~ x + w', data=df2).fit()
print("=== Q2 (d) OLS: y ~ x + w ===")
print(fit_d.summary())
print("95% CI for slope of x:", fit_d.conf_int().loc['x'].round(3).tolist(), "\n")

# ---------- 5) (e) VIF for x and w (using the actual design matrix) ----------
X = fit_d.model.exog
names = fit_d.model.exog_names
print("=== Q2 (e) VIF (including intercept) ===")
for i, name in enumerate(names):
    vif_val = variance_inflation_factor(X, i)
    print(f"{name:>12s}: {vif_val:6.3f}")
print()


# Q3
# ---------- 1) Generate data ----------
np.random.seed(0)
n = 500
x = np.random.uniform(0, 5, n)
y = x + np.random.normal(0, 1, n)
w = 4 + 2*x + 3*y + np.random.normal(0, 1, n)

df3 = pd.DataFrame({'x': x, 'y': y, 'w': w})

# ---------- 2) (b) Correlation and basic descriptive stats ----------
print("=== Q3 (b) Correlation matrix ===")
print(df3.corr().round(3), "\n")

print("=== Q3 (b) Descriptive stats (min, max, mean, std) ===")
print(df3.describe().loc[['min', 'max', 'mean', 'std']].round(3), "\n")

# ---------- 3) (c) OLS: y ~ x ----------
fit1 = smf.ols('y ~ x', data=df3).fit()
print("=== Q3 (c) OLS: y ~ x ===")
print(fit1.summary())
print("95% CI for slope of x:", fit1.conf_int().loc['x'].round(3).tolist())
print("R^2 (y ~ x):", round(fit1.rsquared, 3), "\n")

# ---------- 4) (d) OLS: y ~ x + w ----------
fit2 = smf.ols('y ~ x + w', data=df3).fit()
print("=== Q3 (d) OLS: y ~ x + w ===")
print(fit2.summary())
print("95% CI for slope of x:", fit2.conf_int().loc['x'].round(3).tolist())
print("R^2 (y ~ x + w):", round(fit2.rsquared, 3), "\n")

# ---------- 5) (e) VIF ----------
X = fit2.model.exog
names = fit2.model.exog_names
print("=== Q3 (e) VIF (including intercept) ===")
for i, name in enumerate(names):
    vif_val = variance_inflation_factor(X, i)
    print(f"{name:>12s}: {vif_val:6.3f}")
print()

# ---------- 6) (f) Note for write-up ----------
"""
Interpretation note (to include in your PDF write-up, not code):
- Controlling a collider (w) opens a spurious path between x and y.
- As a result, the slope on x in y~x+w becomes biased (often changes sign or drifts),
  even though R^2 typically increases.
- Higher R^2 does not imply the model is the correct *causal* specification.
"""

# Q4
# 1) Generate data
np.random.seed(0)
n = 500
x = np.random.uniform(0, 5, n)
w = x + np.random.normal(0, 1, n)
y = 2*w + np.random.normal(0, 1, n)

df4 = pd.DataFrame({'x': x, 'w': w, 'y': y})

# 2) Correlation and basic stats
print("=== Q4 (b) Correlation matrix ===")
print(df4.corr().round(3), "\n")

print("=== Q4 (b) Descriptive stats (min, max, mean, std) ===")
print(df4.describe().loc[['min','max','mean','std']].round(3), "\n")

# 3) y ~ x (total effect of x on y through w)
m1 = smf.ols('y ~ x', data=df4).fit()
print("=== Q4 (c) OLS: y ~ x ===")
print(m1.summary())
print("95% CI for slope of x:", m1.conf_int().loc['x'].round(3).tolist(), "\n")

# 4) y ~ x + w (controls mediator; x should be ~0 and not significant)
m2 = smf.ols('y ~ x + w', data=df4).fit()
print("=== Q4 (d) OLS: y ~ x + w ===")
print(m2.summary())
print("95% CI for slope of x:", m2.conf_int().loc['x'].round(3).tolist(), "\n")

# 5) R^2 comparison
print("R^2 (y ~ x)    :", round(m1.rsquared, 3))
print("R^2 (y ~ x + w):", round(m2.rsquared, 3))
# Note: R^2 increases after adding w, but that does not mean it's the "right" causal model
# if the target is the TOTAL effect of x on y (through mediator w).


# Q5
# 1) Generate data (same seed as spec)
np.random.seed(1)
x1 = np.random.rand(100)
x2 = 0.5*x1 + np.random.normal(0, 0.1, 100)  # sd=0.1
y  = 2 + 2*x1 + 0.3*x2 + np.random.normal(0, 1, 100)

df5 = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})

print("=== Q5 (a) True model ===")
print("y = 2 + 2*x1 + 0.3*x2 + e,  sd(e)=1\n")

# 2) corr(x1,x2)
print("=== Q5 (b) corr(x1,x2) ===")
print(round(df5['x1'].corr(df5['x2']), 3), "\n")

# 3) y ~ x1 + x2
m_full = smf.ols('y ~ x1 + x2', data=df5).fit()
print("=== Q5 (c) OLS: y ~ x1 + x2 ===")
print(m_full.summary())
print("95% CI:\n", m_full.conf_int().round(3), "\n")

# 4) y ~ x1
m_x1 = smf.ols('y ~ x1', data=df5).fit()
print("=== Q5 (d) OLS: y ~ x1 ===")
print(m_x1.summary())
print("95% CI:\n", m_x1.conf_int().round(3), "\n")

# 5) y ~ x2
m_x2 = smf.ols('y ~ x2', data=df5).fit()
print("=== Q5 (e) OLS: y ~ x2 ===")
print(m_x2.summary())
print("95% CI:\n", m_x2.conf_int().round(3), "\n")

# 6) Notes for (f) in your write-up (no code):
"""
Interpretation notes for part (f):
- x1 and x2 are highly correlated, so single-predictor models suffer omitted-variable bias:
  y~x1 absorbs some effect of x2; y~x2 absorbs some effect of x1.
- In the full model y~x1+x2, collinearity inflates standard errors and destabilizes coefficients.
- Therefore the results across (c)–(e) may look inconsistent but are not contradictory;
  they reflect collinearity and omitted-variable bias mechanics.
"""