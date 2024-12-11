import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv("./class_project/data/csv/kbo_pitchers.csv")

data = raw_data.copy()
data.dropna(inplace=True)
data.rename(columns={"Age 7/1/24": "Age"}, inplace=True)

cpi = {
    2010: 90.0,
    2011: 94.0,
    2012: 96.0,
    2013: 97.0,
    2014: 98.0,
    2015: 99.0,
    2016: 100.0,
    2017: 101.0,
    2018: 103.0,
    2019: 104.0,
    2020: 105.0,
}

after_NC = {
    2010: 0,
    2011: 0,
    2012: 0,
    2013: 1,
    2014: 1,
    2015: 1,
    2016: 1,
    2017: 1,
    2018: 1,
    2019: 1,
    2020: 1,
}

after_KT = {
    2010: 0,
    2011: 0,
    2012: 0,
    2013: 0,
    2014: 0,
    2015: 1,
    2016: 1,
    2017: 1,
    2018: 1,
    2019: 1,
    2020: 1,
}


data["cpi"] = data["year"].map(cpi)
data["cpi_rate"] = data["cpi"] / cpi[2010]

# model is similar to Cobb-Douglas production function
data["AAV"] = np.log(data["AAV"] / data["cpi_rate"])
# data["after_NC"] = data["year"].map(after_NC)
# data["after_KT"] = data["year"].map(after_KT)

# drop unnecessary columns
data.drop(
    [
        "Player", 
        "cpi", 
        "year", 
        "cpi_rate",
        "Old Club", 
        "New Club",
    ], axis=1, inplace=True
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
plt.savefig("./class_project/images/kbo_pitchers_fitted.png")
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
plt.savefig("./class_project/images/kbo_pitchers_residuals.png")
# plt.show()