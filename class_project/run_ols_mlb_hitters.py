import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv("./class_project/data/csv/mlb_hitters.csv")

data = raw_data.copy()
data.dropna(inplace=True)
data.rename(columns={"Age 7/1/24": "Age"}, inplace=True)

cpi = {
    2010: 218.056,
    2011: 224.939,
    2012: 229.594,
    2013: 232.957,
    2014: 236.736,
    2015: 237.017,
    2016: 240.007,
    2017: 245.120,
    2018: 251.107,
    2019: 255.657,
    2020: 258.811,
}

# adjust for inflation
data["cpi"] = data["year"].map(cpi)
data["cpi_rate"] = data["cpi"] / cpi[2010]
data["AAV"] = np.log(data["AAV"] / data["cpi_rate"])
data["New_Team_Payroll_Prev_Year"] = (
    data["New_Team_Payroll_Prev_Year"] / data["cpi_rate"]
)

# drop unnecessary columns
data.drop(
    [
        "Player",
        "cpi",
        "year",
        "cpi_rate",
        "Old Club",
        "New Club",
        "OBP_league",
        "SLG_league",
    ],
    axis=1,
    inplace=True,
)


# regression
X = data.drop("AAV", axis=1)
X = sm.add_constant(X)
y = data["AAV"]

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

# Residuals and fitted values
residuals = results.resid
fitted = results.fittedvalues

# Plot actual vs fitted
plt.figure(figsize=(8, 6))
plt.scatter(y, fitted)
plt.xlabel("Actual AAV")
plt.ylabel("Fitted AAV")
plt.title("Scatter Plot of Actual vs. Fitted AAV")
plt.savefig("./class_project/images/mlb_hitters_fitted.png")
# plt.show()

# plot residuals versus independent variables
num_vars = len(X.columns) - 1
cols = 4
rows = (num_vars // cols) + (num_vars % cols > 0)

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
axes = axes.flatten()

for i, col in enumerate(X.columns[1:]):
    sns.scatterplot(x=data[col], y=residuals, ax=axes[i])
    axes[i].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[i].set_title(f"Residuals vs. {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Residuals")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("./class_project/images/mlb_hitters_residuals.png")
# plt.show()
