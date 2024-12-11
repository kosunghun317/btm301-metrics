import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns

raw_data = pd.read_csv("data/csv/mlb_pitchers.csv")

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

# model is similar to Cobb-Douglas production function
data["AAV"] = np.log(data["AAV"] / data["cpi_rate"])
data["Win_Pct"] = np.log(data["Win_Pct"])
data["Attendance"] = np.log(data["Attendance"])
data["Age"] = np.log(data["Age"])
data["Is_Left_Handed"] += 1
data["New_Team_Payroll_Prev_Year"] = np.log(data["New_Team_Payroll_Prev_Year"])
columns_to_check = ['ERA', 'WHIP', 'SO', 'IP']
data = data.loc[~(data[columns_to_check] == 0).any(axis=1)]
data["ERA"] = np.log(data["ERA"])
data["WHIP"] = np.log(data["WHIP"])
data["SO"] = np.log(data["SO"] + 1)
data["IP"] = np.log(data["IP"] + 1)
data["ERA_league"] = np.log(data["ERA_league"])
data["WHIP_league"] = np.log(data["WHIP_league"])