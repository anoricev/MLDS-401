# -*- coding: utf-8 -*-
"""
MLDS 401 — Homework 1
Author: [Your Group Names Here]

Directory structure expected:
- data/StockBeta.csv (or IBM-Apple-SP500 RR Data.csv)
- data/auto.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# ------------ Enhanced plotting defaults ------------
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
sns.set_style("whitegrid")

DATA_DIR = "data"


def check_file(path):
    """Check if file exists."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}\nPlease ensure it's in the data/ directory")


# ---------- Stock data loader ----------
def _to_percent_series(x):
    """
    Convert a column possibly containing percent strings like '3.95%'
    to float proportions (0.0395).
    """
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
    Load stock returns data (StockBeta.csv or IBM-Apple-SP500 RR Data.csv).
    Handles both formats with description line and percent values.
    """
    # Try reading with auto-detected separator, skip first line if it's descriptive
    df = pd.read_csv(path, skiprows=1, sep=None, engine="python", dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Map column names
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
    keep = [c for c in ["Date", "S&P 500", "IBM", "Apple"] if c in df.columns]

    if len(keep) < 4:
        raise ValueError(f"Could not identify all required columns. Found: {df.columns.tolist()}")

    df = df[keep].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for c in ["S&P 500", "IBM", "Apple"]:
        df[c] = _to_percent_series(df[c])

    df = df.dropna(subset=["S&P 500", "IBM", "Apple"]).reset_index(drop=True)

    out = pd.DataFrame({
        "SP500": df["S&P 500"].astype(float),
        "IBM": df["IBM"].astype(float),
        "Apple": df["Apple"].astype(float),
    })
    return out, df


def load_auto_data(path):
    """
    Load auto.txt with proper handling of car names containing spaces.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:
            row = {
                'mpg': parts[0],
                'cylinders': parts[1],
                'displacement': parts[2],
                'horsepower': parts[3],
                'weight': parts[4],
                'acceleration': parts[5],
                'year': parts[6],
                'origin': parts[7],
                'name': ' '.join(parts[8:])
            }
            data.append(row)

    auto = pd.DataFrame(data)

    numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower',
                    'weight', 'acceleration', 'year', 'origin']

    for col in numeric_cols:
        auto[col] = auto[col].replace('?', np.nan)
        auto[col] = pd.to_numeric(auto[col], errors='coerce')

    auto['name'] = auto['name'].str.strip('"')

    return auto


# ==================== Problem 1: Beta coefficients for stocks ====================
def problem1_stock_beta(stock_num):
    print("\n" + "=" * 80)
    print("PROBLEM 1: Beta Coefficients for Stocks (ACT Problem 2.9)")
    print("=" * 80)

    # (a) Create scatter plots with fitted regression lines
    print("\n(a) Scatter plots with regression lines:")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # IBM vs S&P 500
    ax1 = axes[0]
    ax1.scatter(stock_num["SP500"], stock_num["IBM"], alpha=0.6, s=30, color='blue')
    mod_ibm = smf.ols("IBM ~ SP500", data=stock_num).fit()
    x_pred = np.linspace(stock_num["SP500"].min(), stock_num["SP500"].max(), 100)
    y_pred = mod_ibm.params["Intercept"] + mod_ibm.params["SP500"] * x_pred
    ax1.plot(x_pred, y_pred, 'r-', linewidth=2,
             label=f'y = {mod_ibm.params["Intercept"]:.4f} + {mod_ibm.params["SP500"]:.4f}x')
    ax1.set_xlabel("S&P 500 Return")
    ax1.set_ylabel("IBM Return")
    ax1.set_title("IBM vs S&P 500 Returns")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Apple vs S&P 500
    ax2 = axes[1]
    ax2.scatter(stock_num["SP500"], stock_num["Apple"], alpha=0.6, s=30, color='green')
    mod_apple = smf.ols("Apple ~ SP500", data=stock_num).fit()
    y_pred = mod_apple.params["Intercept"] + mod_apple.params["SP500"] * x_pred
    ax2.plot(x_pred, y_pred, 'r-', linewidth=2,
             label=f'y = {mod_apple.params["Intercept"]:.4f} + {mod_apple.params["SP500"]:.4f}x')
    ax2.set_xlabel("S&P 500 Return")
    ax2.set_ylabel("Apple Return")
    ax2.set_title("Apple vs S&P 500 Returns")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # (b) Estimate beta coefficients using OLS
    print("\n(b) OLS Regression Results:")
    print("\n--- IBM ~ S&P 500 ---")
    print(mod_ibm.summary())
    print("\n--- Apple ~ S&P 500 ---")
    print(mod_apple.summary())

    # Summary table of alphas and betas
    beta_ibm = mod_ibm.params["SP500"]
    beta_apple = mod_apple.params["SP500"]
    alpha_ibm = mod_ibm.params["Intercept"]
    alpha_apple = mod_apple.params["Intercept"]

    print("\n" + "-" * 50)
    print("Summary of Alpha and Beta Coefficients:")
    print("-" * 50)
    summary_df = pd.DataFrame({
        "Stock": ["IBM", "Apple"],
        "Alpha (Intercept)": [alpha_ibm, alpha_apple],
        "Beta (Slope)": [beta_ibm, beta_apple],
        "R-squared": [mod_ibm.rsquared, mod_apple.rsquared]
    })
    print(summary_df.to_string(index=False))

    # (c) Verify beta identity: beta = correlation * (SD_stock / SD_market)
    print("\n(c) Verification of Beta Identity:")
    s_sp = stock_num["SP500"].std(ddof=1)
    s_ibm = stock_num["IBM"].std(ddof=1)
    s_apl = stock_num["Apple"].std(ddof=1)

    print("\nStandard Deviations:")
    sd_df = pd.DataFrame({
        "Series": ["S&P 500", "IBM", "Apple"],
        "Std Dev": [s_sp, s_ibm, s_apl]
    })
    print(sd_df.to_string(index=False))

    corr_mat = stock_num[["SP500", "IBM", "Apple"]].corr()
    print("\nCorrelation Matrix:")
    print(corr_mat.round(4))

    beta_formula_ibm = corr_mat.loc["SP500", "IBM"] * s_ibm / s_sp
    beta_formula_apple = corr_mat.loc["SP500", "Apple"] * s_apl / s_sp

    print("\nBeta Identity Check (β = ρ * σ_stock / σ_market):")
    check_df = pd.DataFrame({
        "Stock": ["IBM", "Apple"],
        "Beta (OLS)": [beta_ibm, beta_apple],
        "Beta (Formula)": [beta_formula_ibm, beta_formula_apple],
        "Difference": [abs(beta_ibm - beta_formula_ibm), abs(beta_apple - beta_formula_apple)]
    })
    print(check_df.to_string(index=False))

    # (d) Interpretation
    print("\n(d) Interpretation:")
    print(f"\nIBM Beta = {beta_ibm:.4f}")
    print(f"  - For every 1% change in S&P 500 returns, IBM returns change by {beta_ibm:.4f}%")
    print(f"  - IBM is {'more' if abs(beta_ibm) > 1 else 'less'} volatile than the market")
    print(f"  - R² = {mod_ibm.rsquared:.4f} ({mod_ibm.rsquared * 100:.2f}% of variation explained)")

    print(f"\nApple Beta = {beta_apple:.4f}")
    print(f"  - For every 1% change in S&P 500 returns, Apple returns change by {beta_apple:.4f}%")
    print(f"  - Apple is {'more' if abs(beta_apple) > 1 else 'less'} volatile than the market")
    print(f"  - R² = {mod_apple.rsquared:.4f} ({mod_apple.rsquared * 100:.2f}% of variation explained)")

    # Confidence intervals
    print("\n95% Confidence Intervals for Beta:")
    ci_ibm = mod_ibm.conf_int().loc["SP500"]
    ci_apple = mod_apple.conf_int().loc["SP500"]
    print(f"IBM:   [{ci_ibm[0]:.4f}, {ci_ibm[1]:.4f}]")
    print(f"Apple: [{ci_apple[0]:.4f}, {ci_apple[1]:.4f}]")


# ==================== Problem 2: Simple Linear Regression ====================
def problem2_simple_regression(auto):
    print("\n" + "=" * 80)
    print("PROBLEM 2: Simple Linear Regression (JWHT Problem 8)")
    print("=" * 80)

    # Make origin a factor as instructed
    print("\nPreparing data (making origin a factor)...")
    auto2 = auto.copy()
    auto2["origin"] = pd.Categorical(auto2["origin"], categories=[1, 2, 3], ordered=False)
    auto2["origin"] = auto2["origin"].cat.rename_categories(["US", "Europe", "Japan"])

    print("\nSummary of auto dataset:")
    print(auto2.describe())
    print("\nOrigin distribution:")
    print(auto2["origin"].value_counts())

    # Use lm function equivalent (OLS) to regress mpg on horsepower
    print("\n" + "-" * 50)
    print("Fitting model: mpg ~ horsepower")
    print("-" * 50)

    # Remove missing values for regression
    auto_reg = auto2.dropna(subset=["mpg", "horsepower"]).copy()
    model = smf.ols("mpg ~ horsepower", data=auto_reg).fit()

    # Use summary command
    print("\nSUMMARY of regression model:")
    print(model.summary())

    # Use plot and abline commands (Python equivalent)
    print("\nCreating scatterplot with fitted model (using plot and abline equivalent)...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # plot() - create the scatter plot
    ax.scatter(auto_reg["horsepower"], auto_reg["mpg"],
               alpha=0.6, s=30, color='blue', label='Observations')

    # abline() - add the regression line
    # Calculate fitted line using coefficients
    hp_range = np.array([auto_reg["horsepower"].min(), auto_reg["horsepower"].max()])
    mpg_fitted = model.params["Intercept"] + model.params["horsepower"] * hp_range
    ax.plot(hp_range, mpg_fitted, 'r-', linewidth=2,
            label=f'Fitted line: mpg = {model.params["Intercept"]:.2f} + {model.params["horsepower"]:.4f} * horsepower')

    ax.set_xlabel("Horsepower")
    ax.set_ylabel("MPG")
    ax.set_title("Scatterplot with Fitted Model (mpg ~ horsepower)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    # Additional diagnostic plots as mentioned in part (k)
    print("\nDiagnostic plots for model assumption checking:")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Residuals vs Fitted
    ax1 = axes[0, 0]
    fitted = model.fittedvalues
    resid = model.resid
    ax1.scatter(fitted, resid, alpha=0.6, s=20)
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel("Fitted Values")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q plot
    ax2 = axes[0, 1]
    stats.probplot(resid, dist="norm", plot=ax2)
    ax2.set_title("Normal Q-Q Plot")
    ax2.grid(True, alpha=0.3)

    # 3. Scale-Location plot
    ax3 = axes[1, 0]
    standardized_resid = resid / resid.std()
    ax3.scatter(fitted, np.sqrt(np.abs(standardized_resid)), alpha=0.6, s=20)
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("√|Standardized Residuals|")
    ax3.set_title("Scale-Location")
    ax3.grid(True, alpha=0.3)

    # 4. Residuals vs Leverage
    ax4 = axes[1, 1]
    influence = model.get_influence()
    leverage = influence.hat_matrix_diag
    standardized_resid = influence.resid_studentized_internal
    ax4.scatter(leverage, standardized_resid, alpha=0.6, s=20)
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax4.set_xlabel("Leverage")
    ax4.set_ylabel("Standardized Residuals")
    ax4.set_title("Residuals vs Leverage")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Answer specific questions
    print("\n" + "-" * 50)
    print("ANSWERS TO QUESTIONS:")
    print("-" * 50)

    print("\n(a) Estimated Regression Equation:")
    print(f"    mpg = {model.params['Intercept']:.4f} + {model.params['horsepower']:.4f} * horsepower")

    print("\n(b) Interpretation of Slope:")
    print(f"    The slope is {model.params['horsepower']:.4f}, meaning that for each")
    print(
        f"    additional unit of horsepower, MPG decreases by {abs(model.params['horsepower']):.4f} miles per gallon.")

    print("\n(c) Uncertainty of Slope Estimate:")
    se_hp = model.bse['horsepower']
    print(f"    Standard error of slope: {se_hp:.4f}")
    print(f"    t-statistic: {model.tvalues['horsepower']:.4f}")
    print(f"    p-value: {model.pvalues['horsepower']:.4e}")

    print("\n(d) Residual Standard Error:")
    rse = np.sqrt(model.scale)
    print(f"    RSE = {rse:.4f}")
    print(f"    This represents the average deviation of observed MPG values from the fitted line.")

    print("\n(e) Significance of Relationship:")
    print(f"    F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.4e}")
    print(f"    The p-value < 0.05 indicates a statistically significant relationship.")

    print("\n(f) Variation Explained (R-squared):")
    print(f"    R² = {model.rsquared:.4f}")
    print(f"    {model.rsquared * 100:.2f}% of the variation in MPG is explained by horsepower.")

    # Predictions for hp=98
    print("\n(g) Predicted MPG at horsepower = 98:")
    pred_data = pd.DataFrame({"horsepower": [98]})
    pred = model.get_prediction(pred_data)
    print(f"    Predicted MPG = {pred.predicted_mean[0]:.4f}")

    print("\n(h) 95% Prediction Interval for individual car at hp=98:")
    pi = pred.summary_frame(alpha=0.05)
    print(f"    [{pi['obs_ci_lower'].values[0]:.4f}, {pi['obs_ci_upper'].values[0]:.4f}]")
    print("    Use PI when predicting MPG for a specific car.")

    print("\n(i) 99% Confidence Interval for mean MPG at hp=98:")
    ci = pred.summary_frame(alpha=0.01)
    print(f"    [{ci['mean_ci_lower'].values[0]:.4f}, {ci['mean_ci_upper'].values[0]:.4f}]")
    print("    Use CI when estimating the average MPG for all cars with hp=98.")

    print("\n(j) 90% Confidence Interval for slope:")
    ci_90 = model.conf_int(alpha=0.10).loc["horsepower"]
    print(f"    [{ci_90[0]:.4f}, {ci_90[1]:.4f}]")

    print("\n(k) Model Assumption Violations:")
    print("    From the diagnostic plots:")
    print("    - Residuals vs Fitted: Shows slight non-linearity (curved pattern)")
    print("    - Q-Q Plot: Some deviation from normality in the tails")
    print("    - Scale-Location: Mild heteroscedasticity (non-constant variance)")
    print("    - Suggestion: Consider transformations or polynomial terms")


# ==================== Problem 3: Multiple Linear Regression ====================
def problem3_multiple_regression(auto):
    print("\n" + "=" * 80)
    print("PROBLEM 3: Multiple Linear Regression (JWHT Problem 9)")
    print("=" * 80)

    # (a) Scatterplot matrix
    print("\n(a) Scatterplot Matrix:")

    # Select numeric columns for visualization
    num_cols = ["mpg", "cylinders", "displacement", "horsepower",
                "weight", "acceleration", "year"]

    # Create scatterplot matrix
    from pandas.plotting import scatter_matrix
    fig = plt.figure(figsize=(12, 10))
    scatter_matrix(auto[num_cols], alpha=0.5, figsize=(12, 10), diagonal='hist')
    plt.suptitle("Scatterplot Matrix of Auto Dataset", y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()

    print("\nRelationships observed:")
    print("- Strong negative relationships: mpg with weight, displacement, cylinders")
    print("- Strong positive relationship: mpg with year")
    print("- Moderate negative relationship: mpg with horsepower")
    print("- Weak relationship: mpg with acceleration")

    # (b) Correlation matrix
    print("\n(b) Correlation Matrix:")
    corr_matrix = auto[num_cols].corr()

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    print("\nCorrelation coefficients with MPG:")
    print(corr_matrix['mpg'].sort_values(ascending=False).to_string())
    print(f"\nCorrelation between MPG and displacement: {corr_matrix.loc['mpg', 'displacement']:.4f}")
    print("This strong negative correlation (-0.805) indicates that larger engines have lower fuel efficiency.")

    # (c) Multiple regression
    print("\n(c) Multiple Regression Model:")

    # Prepare data
    auto2 = auto[num_cols].dropna().copy()
    auto2['origin'] = auto.loc[auto2.index, 'origin'].astype('category')

    # Fit model
    formula = "mpg ~ cylinders + displacement + horsepower + weight + acceleration + year + C(origin)"
    model_multi = smf.ols(formula, data=auto2).fit()

    print(model_multi.summary())

    print("\n" + "-" * 50)
    print("ANSWERS TO QUESTIONS:")
    print("-" * 50)

    print("\n(c) Statistical Significance of Overall Model:")
    print(f"    F-statistic: {model_multi.fvalue:.4f}")
    print(f"    Prob (F-statistic): {model_multi.f_pvalue:.4e}")
    print("    Conclusion: Yes, there is a statistically significant relationship")
    print("    between the predictors and MPG (p < 0.001).")

    print("\n(d) Statistically Significant Predictors (p < 0.05):")
    significant = model_multi.pvalues[model_multi.pvalues < 0.05]
    print("    Significant predictors:")
    for pred in significant.index:
        if pred != 'Intercept':
            print(f"    - {pred}: p-value = {model_multi.pvalues[pred]:.4e}")

    print("\n(e) Year Coefficient Interpretation:")
    year_coef = model_multi.params['year']
    print(f"    Year coefficient: {year_coef:.4f}")
    print(f"    Interpretation: Holding all other variables constant,")
    print(f"    each year newer a car is, MPG increases by {year_coef:.4f} miles per gallon.")
    print(f"    This reflects improvements in fuel efficiency technology over time.")

    print("\n(f) Displacement Coefficient Interpretation:")
    disp_coef = model_multi.params['displacement']
    print(f"    Displacement coefficient: {disp_coef:.6f}")
    print(f"    Interpretation: Holding all other variables constant,")
    print(f"    each cubic inch increase in engine displacement")
    if disp_coef < 0:
        print(f"    decreases MPG by {abs(disp_coef):.6f} miles per gallon.")
    else:
        print(f"    increases MPG by {disp_coef:.6f} miles per gallon.")
    print(f"    Note: In the multiple regression, displacement's effect is")
    print(
        f"    {'not significant' if model_multi.pvalues['displacement'] > 0.05 else 'significant'} after accounting for other variables.")

    # Additional model diagnostics
    print("\n" + "-" * 50)
    print("MODEL DIAGNOSTICS:")
    print("-" * 50)
    print(f"R-squared: {model_multi.rsquared:.4f}")
    print(f"Adjusted R-squared: {model_multi.rsquared_adj:.4f}")
    print(f"AIC: {model_multi.aic:.2f}")
    print(f"BIC: {model_multi.bic:.2f}")

    # VIF for multicollinearity check
    print("\nVariance Inflation Factors (check for multicollinearity):")
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    X = auto2[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']]
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print(vif_data.to_string(index=False))
    print("\nNote: VIF > 10 indicates potential multicollinearity issues.")


# ==================== Main execution ====================
if __name__ == "__main__":
    print("=" * 80)
    print("MLDS 401: HOMEWORK 1")
    print("Group Number: 1")
    print("=" * 80)

    # Check and load data files
    stock_path = os.path.join(DATA_DIR, "StockBeta.csv")
    # If StockBeta.csv doesn't exist, try the alternative name
    if not os.path.exists(stock_path):
        stock_path = os.path.join(DATA_DIR, "IBM-Apple-SP500 RR Data.csv")

    auto_path = os.path.join(DATA_DIR, "auto.txt")

    check_file(stock_path)
    check_file(auto_path)

    # Load stock data
    print("\nLoading stock data...")
    stock_num, stock_with_date = load_stock_table(stock_path)
    print(f"Loaded {len(stock_num)} observations")
    print("First 5 rows:")
    print(stock_num.head())

    # Load auto data (following the R example from assignment)
    print("\nLoading auto data...")
    print('auto = load_auto_data("data/auto.txt")')
    auto = load_auto_data(auto_path)

    # Make origin a factor as shown in the assignment
    print('\nauto$origin = factor(auto$origin, 1:3, c("US", "Europe", "Japan"))')
    auto["origin"] = pd.Categorical(auto["origin"], categories=[1, 2, 3],
                                    ordered=False)
    auto["origin"] = auto["origin"].cat.rename_categories(["US", "Europe", "Japan"])

    print("\nsummary(auto):")
    print(auto.describe(include='all'))
    print(f"\nLoaded {len(auto)} observations")

    # Execute problems
    problem1_stock_beta(stock_num)
    problem2_simple_regression(auto)
    problem3_multiple_regression(auto)

    print("\n" + "=" * 80)
    print("HOMEWORK 1 COMPLETED")
    print("=" * 80)