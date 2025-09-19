# -*- coding: utf-8 -*-
"""
MLDS 401 — Homework 1 (Python)
Author: Group 1

Directory structure expected:
- data/IBM-Apple-SP500 RR Data.csv
- data/auto.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

# ------------ basic plotting defaults ------------
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 150

DATA_DIR = "data"


def check_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path} ；请确认放在项目下的 data/ 目录")


# ---------- loader for the stock CSV with a description line ----------
def _to_percent_series(x):
    """
    Convert a column possibly containing percent strings like '3.95%'
    to float proportions (0.0395). Non-conforming entries become NaN
    and are left to dropna later.
    """
    # already numeric?
    if np.issubdtype(pd.Series(x).dtype, np.number):
        return pd.to_numeric(x, errors="coerce")
    s = (pd.Series(x)
         .astype(str)
         .str.strip()
         .str.replace("%", "", regex=False)
         .str.replace(",", "", regex=False))
    return pd.to_numeric(s, errors="coerce") / 100.0


def load_stock_table(path):
    """
    Robust reader for the monthly return table that starts with
    a description line ('Monthly Return Rate* ...'), followed by
    a header row: Date, S&P 500, IBM, Apple, then rows with percent values.
    Works with tab- or comma-separated files.
    """
    # try to auto-detect separator, skip the first line
    df = pd.read_csv(path, skiprows=1, sep=None, engine="python", dtype=str)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    # standardize expected column names
    # some files may have trailing empty column
    want = ["Date", "S&P 500", "IBM", "Apple"]
    # try to locate closest names
    col_map = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ("date",):
            col_map[c] = "Date"
        elif "s&p" in lc or "sp 500" in lc or "s&p 500" in lc:
            col_map[c] = "S&P 500"
        elif "ibm" in lc:
            col_map[c] = "IBM"
        elif "apple" in lc or "aapl" in lc:
            col_map[c] = "Apple"
    df = df.rename(columns=col_map)
    # keep only needed columns if present
    keep = [c for c in want if c in df.columns]
    if len(keep) < 4:
        raise ValueError(f"未能识别到完整列（期望列：{want}；当前列：{df.columns.tolist()}）")
    df = df[keep].copy()

    # parse date
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # convert percent columns to numeric proportions
    for c in ["S&P 500", "IBM", "Apple"]:
        df[c] = _to_percent_series(df[c])

    # drop rows with any NaNs in numeric columns
    df = df.dropna(subset=["S&P 500", "IBM", "Apple"]).reset_index(drop=True)

    # final numeric frame for modeling
    out = pd.DataFrame({
        "SP500": df["S&P 500"].astype(float),
        "IBM": df["IBM"].astype(float),
        "Apple": df["Apple"].astype(float),
    })
    return out, df  # (numeric_for_model, original_with_date)


def load_auto_data(path):
    """
    Load the auto.txt file with proper handling of car names that may contain spaces.
    The file has columns: mpg cylinders displacement horsepower weight acceleration
    year origin name
    """
    # First, try to read the file line by line to understand its structure
    with open(path, 'r') as f:
        lines = f.readlines()

    # Parse each line manually
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:  # We need at least 9 parts
            # First 8 fields are numeric/categorical
            # Everything after the 8th field is the car name
            row = {
                'mpg': parts[0],
                'cylinders': parts[1],
                'displacement': parts[2],
                'horsepower': parts[3],
                'weight': parts[4],
                'acceleration': parts[5],
                'year': parts[6],
                'origin': parts[7],
                'name': ' '.join(parts[8:])  # Join all remaining parts as the name
            }
            data.append(row)

    # Create DataFrame
    auto = pd.DataFrame(data)

    # Convert numeric columns
    numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower',
                    'weight', 'acceleration', 'year', 'origin']

    for col in numeric_cols:
        # Replace '?' with NaN and convert to numeric
        auto[col] = auto[col].replace('?', np.nan)
        auto[col] = pd.to_numeric(auto[col], errors='coerce')

    # Clean up the name column (remove quotes if present)
    auto['name'] = auto['name'].str.strip('"')

    return auto


# -------------------- Q1: Market Model --------------------
def scatter_with_fit(x, y, xlab, ylab, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.7)
    # simple OLS line using numpy
    beta1, beta0 = np.polyfit(x, y, deg=1)  # slope, intercept
    xgrid = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    yhat = beta1 * xgrid + beta0
    ax.plot(xgrid, yhat)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return beta0, beta1


def q1(stock_num):
    print("\n=== Q1: Stock Beta (Market Model) ===")
    # (a) scatter plots
    scatter_with_fit(stock_num["SP500"], stock_num["IBM"],
                     "S&P 500 Return", "IBM Return", "IBM vs S&P 500")
    scatter_with_fit(stock_num["SP500"], stock_num["Apple"],
                     "S&P 500 Return", "Apple Return", "Apple vs S&P 500")

    # (b) OLS betas
    mod_ibm = smf.ols("IBM ~ SP500", data=stock_num).fit()
    mod_apple = smf.ols("Apple ~ SP500", data=stock_num).fit()
    print("\n--- OLS: IBM ~ SP500 ---")
    print(mod_ibm.summary())
    print("\n--- OLS: Apple ~ SP500 ---")
    print(mod_apple.summary())

    beta_ibm = mod_ibm.params["SP500"]
    beta_apple = mod_apple.params["SP500"]
    print("\nAlpha/Beta table:")
    print(pd.DataFrame({
        "Stock": ["IBM", "Apple"],
        "Alpha": [mod_ibm.params["Intercept"], mod_apple.params["Intercept"]],
        "Beta": [beta_ibm, beta_apple]
    }))

    # (c) SDs, correlations, beta identity
    s_sp = stock_num["SP500"].std(ddof=1)
    s_ibm = stock_num["IBM"].std(ddof=1)
    s_apl = stock_num["Apple"].std(ddof=1)
    print("\nSample SDs:")
    print(pd.DataFrame({"Series": ["S&P 500", "IBM", "Apple"],
                        "SD": [s_sp, s_ibm, s_apl]}))
    corr_mat = stock_num[["SP500", "IBM", "Apple"]].corr()
    print("\nCorrelation matrix:")
    print(corr_mat)

    beta_by_id_ibm = corr_mat.loc["SP500", "IBM"] * s_ibm / s_sp
    beta_by_id_apple = corr_mat.loc["SP500", "Apple"] * s_apl / s_sp
    check = pd.DataFrame({
        "Stock": ["IBM", "Apple"],
        "Beta_from_OLS": [beta_ibm, beta_apple],
        "Beta_r_sy_over_sx": [beta_by_id_ibm, beta_by_id_apple],
        "Diff": [beta_ibm - beta_by_id_ibm, beta_apple - beta_by_id_apple]
    })
    print("\nBeta identity check:")
    print(check)

    # (d) interpretation helpers
    def glance_like(res):
        return pd.Series({
            "nobs": res.nobs,
            "rsquared": res.rsquared,
            "rsquared_adj": res.rsquared_adj,
            "fvalue": res.fvalue,
            "f_pvalue": res.f_pvalue,
            "aic": res.aic,
            "bic": res.bic,
            "se_reg": np.sqrt(res.scale)  # residual std error
        })

    print("\nModel summary (glance-like): IBM")
    print(glance_like(mod_ibm))
    print("\nModel summary (glance-like): Apple")
    print(glance_like(mod_apple))

    print("\n95% CI for Beta:")
    print("IBM  :", mod_ibm.conf_int().loc["SP500"].to_list())
    print("Apple:", mod_apple.conf_int().loc["SP500"].to_list())

    print("\nVolatility (SD) vs Beta:")
    print(pd.DataFrame({
        "Stock": ["IBM", "Apple"],
        "SD": [s_ibm, s_apl],
        "Beta": [beta_ibm, beta_apple]
    }))


# -------------------- Q2: Simple Linear Regression --------------------
def q2(auto):
    print("\n=== Q2: mpg ~ horsepower ===")
    # keep rows with both mpg & horsepower
    auto2 = auto.dropna(subset=["mpg", "horsepower"]).copy()

    # scatter + fit line
    fig, ax = plt.subplots()
    ax.scatter(auto2["horsepower"], auto2["mpg"], alpha=0.7)
    m_hp = smf.ols("mpg ~ horsepower", data=auto2).fit()
    hp_grid = np.linspace(auto2["horsepower"].min(), auto2["horsepower"].max(), 200)
    yhat = m_hp.predict(pd.DataFrame({"horsepower": hp_grid}))
    ax.plot(hp_grid, yhat)
    ax.set_xlabel("horsepower")
    ax.set_ylabel("mpg")
    ax.set_title("mpg vs horsepower")
    plt.tight_layout()
    plt.show()

    print(m_hp.summary())
    print("RSE (残差标准差) ≈", np.sqrt(m_hp.scale))
    print("R^2 =", m_hp.rsquared, "  F-stat p-value =", m_hp.f_pvalue)

    # predictions at hp=98
    newd = pd.DataFrame({"horsepower": [98]})
    pred_mean = m_hp.get_prediction(newd)
    sf_mean = pred_mean.summary_frame(alpha=0.01)  # 99% CI for mean
    print("\n99% CI for mean mpg at hp=98:")
    print(sf_mean[["mean", "mean_ci_lower", "mean_ci_upper"]])

    sf_pred = pred_mean.summary_frame(alpha=0.05)  # 95% PI for a single obs
    print("\n95% PI for a single car at hp=98:")
    print(sf_pred[["mean", "obs_ci_lower", "obs_ci_upper"]])

    ci90 = m_hp.conf_int(alpha=0.10).loc["horsepower"].to_list()
    print("90% CI for slope (horsepower):", ci90)

    # simple diagnostics
    fitted = m_hp.fittedvalues
    resid = m_hp.resid

    # Residuals vs Fitted
    fig, ax = plt.subplots()
    ax.scatter(fitted, resid)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    plt.tight_layout()
    plt.show()

    # QQ plot
    sm.ProbPlot(resid).qqplot(line="45")
    plt.title("QQ Plot of Residuals")
    plt.tight_layout()
    plt.show()


# -------------------- Q3: Multiple Linear Regression --------------------
def q3(auto):
    print("\n=== Q3: Multiple Linear Regression & Correlations ===")
    num_cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year"]
    # correlation matrix
    cor_mat = auto[num_cols].corr()
    print("\nCorrelation matrix (numeric columns):")
    print(cor_mat.round(3))
    print("\ncor(mpg, displacement) =", round(cor_mat.loc["mpg", "displacement"], 3))

    # origin as categorical
    auto2 = auto.dropna(subset=num_cols).copy()
    auto2["origin"] = auto2["origin"].astype("category")
    formula = "mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + C(origin)"
    mod_multi = smf.ols(formula, data=auto2).fit()
    print("\n--- OLS: Multiple Regression ---")
    print(mod_multi.summary())


# -------------------- main --------------------
if __name__ == "__main__":
    # files
    stock_path = os.path.join(DATA_DIR, "IBM-Apple-SP500 RR Data.csv")
    auto_path = os.path.join(DATA_DIR, "auto.txt")
    check_file(stock_path)
    check_file(auto_path)

    # Q1 data
    stock_num, stock_with_date = load_stock_table(stock_path)
    print("Loaded stock numeric frame (head):")
    print(stock_num.head())

    # Q2/Q3 data - Use the new loader function
    auto = load_auto_data(auto_path)
    auto["origin"] = auto["origin"].astype("category")
    auto["name"] = auto["name"].astype(str)

    # run tasks
    q1(stock_num)
    q2(auto)
    q3(auto)

    print("\nAll tasks completed.")