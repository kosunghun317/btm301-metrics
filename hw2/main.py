"""
Code for HW2.
For the reproduction, one should delete the last line of nerlove.asc file.
This line is nothing but  character.
Also, nerlove.asc should be in <root of repository>/hw2 directory.
"""

import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


def get_b(y, X):
    """
    get OLS estimator b of unknown beta from given data
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y


def get_R_squared(y, X):
    b = get_b(y, X)
    y_hat = X @ b
    y_bar = np.mean(y)
    SSE = (y - y_hat).T @ (y - y_hat)
    SST = (y - y_bar).T @ (y - y_bar)

    return 1 - SSE / SST


def get_std_errs(y, X):
    """
    get standard errors of each b_i
    """
    b = get_b(y, X)
    N, K = X.shape
    residuals = y - X @ b
    degree_of_freedom = N - K
    sigma_squared = (residuals.T @ residuals) / degree_of_freedom
    variance_of_b = sigma_squared * np.linalg.inv(X.T @ X)
    result = np.sqrt(np.diag(variance_of_b))

    return result


def t_test(y, X):
    """
    get t-test results for each b_i of OLS estimator b
    """
    N, K = X.shape
    b = get_b(y, X).flatten()
    std_errs = get_std_errs(y, X).flatten()
    t_stats = b / std_errs
    p_values = 2 * (
        1 - stats.t.cdf(np.abs(t_stats), N - K)
    )  # p_i = 2 * Prob(t > |t_i|) = 1 - Prob(|t| < |t_i|)

    return t_stats, p_values


def F_test(y, X, R, r):
    """
    F-test for linear restrictions R @ beta = r on the model y = X @ beta + epsilon.
    """
    N, K = X.shape
    b = get_b(y, X)
    residuals = y - X @ b
    degree_of_freedom = N - K
    sigma_squared = (residuals.T @ residuals) / degree_of_freedom
    variance_of_b = sigma_squared * np.linalg.inv(X.T @ X)

    num_of_restrictions = R.shape[0]
    F_stat = (
        (R @ b - r).T
        @ np.linalg.inv(R @ variance_of_b @ R.T)
        @ (R @ b - r)
        / num_of_restrictions
    )  # formula from Hayashi ch.1

    p_value = 1 - stats.f.cdf(F_stat, num_of_restrictions, degree_of_freedom)

    return F_stat, p_value


def plot_residuals(y, X, i, x_label):
    """
    plot residuals of OLS estimator b
    """
    b = get_b(y, X)
    residuals = y - X @ b

    # x-axis: i-th column of X
    # y-axis: residuals
    plt.scatter(X[:, i], residuals)
    plt.axhline(
        y=np.mean(residuals), color="r", linestyle="-", label="mean of residuals"
    )
    plt.xlabel(x_label)
    plt.ylabel("Residuals")
    plt.title(f"Residuals over {x_label}")
    plt.legend()
    plt.savefig(f"hw2/residuals_{x_label}.png")
    plt.show()


data = np.loadtxt("hw2/nerlove.asc")  # 145 x 5 matrix

##############################################################################
#                                 Problem 2                                  #
##############################################################################

print("#" * 34 + " Problem 2 " + "#" * 34)
y = np.log(data[:, 0]).reshape(-1, 1)  # 145 x 1 matrix; log(TC)
X = np.hstack(
    (np.ones((data.shape[0], 1)), np.log(data[:, [1, 2, 3, 4]]))
)  # 145 x 5 matrix; 1, log(Q), log(PL), log(PF), log(PK)

# OLS estimator b
b = get_b(y, X)
print(f"OLS coefficients: \n{b.flatten()}")

# Standard errors
std_errs = get_std_errs(y, X)
print(f"Standard errors: \n{std_errs}")

# t-test results
t_stats, p_values = t_test(y, X)
print(f"t-stats: \n{t_stats}")
print(f"p-values: \n{p_values}")

# R-squared
R_squared = get_R_squared(y, X)
print(f"R-squared: \n{R_squared.flatten()}")

##############################################################################
#                                 Problem 3                                  #
##############################################################################

print("#" * 34 + " Problem 3 " + "#" * 34)
y = np.log(data[:, 0] / data[:, 3])  # 145 x 1 matrix; log(TC / PF)
X = np.hstack(
    (
        np.ones((data.shape[0], 1)),
        np.log(data[:, 1]).reshape(-1, 1),
        np.log(data[:, 2] / data[:, 3]).reshape(-1, 1),
        np.log(data[:, 4] / data[:, 3]).reshape(-1, 1),
    )
)  # 145 x 5 matrix; 1, log(Q), log(PL / PF), log(PK / PF)

# OLS estimator b
b = get_b(y, X)
print(f"OLS coefficients: \n{b.flatten()}")

# Standard errors
std_errs = get_std_errs(y, X)
print(f"Standard errors: \n{std_errs}")

# R-squared
R_squared = get_R_squared(y, X)
print(f"R-squared: \n{R_squared.flatten()}")

##############################################################################
#                                 Problem 4                                  #
##############################################################################

print("#" * 34 + " Problem 4 " + "#" * 34)

# unrestricted model
y = np.log(data[:, 0]).reshape(-1, 1)  # 145 x 1 matrix; log(TC)
X = np.hstack(
    (np.ones((data.shape[0], 1)), np.log(data[:, [1, 2, 3, 4]]))
)  # 145 x 5 matrix; 1, log(Q), log(PL), log(PF), log(PK)

# restrictions
R = np.array([[0, 0, 1, 1, 1]])
r = np.array([[1]])

# F-test
F_stat, p_value = F_test(y, X, R, r)
print(f"F-stat: {F_stat.flatten()}")
print(f"p-value: {p_value.flatten()}")

##############################################################################
#                                 Problem 5                                  #
##############################################################################

print("#" * 34 + " Problem 5 " + "#" * 34)
y = np.log(data[:, 0] / data[:, 3])  # 145 x 1 matrix; log(TC / PF)
X = np.hstack(
    (
        np.ones((data.shape[0], 1)),
        np.log(data[:, 1]).reshape(-1, 1),
        np.log(data[:, 2] / data[:, 3]).reshape(-1, 1),
        np.log(data[:, 4] / data[:, 3]).reshape(-1, 1),
    )
)  # 145 x 5 matrix; 1, log(Q), log(PL / PF), log(PK / PF)

# plot residuals
plot_residuals(y, X, 1, "log(Q)")
