import numpy as np
import scipy.stats as stats


def ols(y, X):
    """
    Implementation of OLS estimator
    """
    N, K = X.shape
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return (N, K, beta)


def std_errors(y, X):
    """
    Calculate standard errors of OLS estimator
    """
    N, K, beta = ols(y, X)
    residual = y - X @ beta
    degree_of_freedom = N - K
    sigma_squared = (residual.T @ residual) / degree_of_freedom
    variance_matrix = sigma_squared * np.linalg.inv(X.T @ X)
    result = np.sqrt(np.diag(variance_matrix))

    return result


def t_test(y, X):
    """
    Perform t-test for each beta of OLS estimator
    """
    N, K, beta = ols(y, X)
    std_err = std_errors(y, X)
    t_stat = beta / std_err
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), N - K))

    return t_stat, p_value


def R_squared(y, X):
    """
    Calculate R-squared of OLS estimator
    """
    N, K, beta = ols(y, X)
    y_hat = X @ beta
    y_bar = np.mean(y)
    SSR = (y_hat - y_bar).T @ (y_hat - y_bar)
    SST = (y - y_bar).T @ (y - y_bar)
    R_squared = SSR / SST

    return R_squared

def F_test(y, X, X_restricted):
    """
    Perform F-test for nested regression models
    """
    N, K, beta = ols(y, X)
    N_restricted, K_restricted, beta_restricted = ols(y, X_restricted)
    SSR = (y - X @ beta).T @ (y - X @ beta)
    SSR_restricted = (y - X_restricted @ beta_restricted).T @ (y - X_restricted @ beta_restricted)
    F_stat = ((SSR_restricted - SSR) / (K - K_restricted)) / (SSR / (N - K))
    p_value = 1 - stats.f.cdf(F_stat, K - K_restricted, N - K)

    return F_stat, p_value


data = np.loadtxt("hw2/nerlove.asc")  # 145 x 5 matrix

##############################################################################
# Problem 2                                                                  #
# log(TC) =                                                                  #
#   beta_1                                                                   #
#   + beta_2 * log(Q)                                                        #
#   + beta_3 * log(PL)                                                       #
#   + beta_4 * log(PF)                                                       #
#   + beta_5 * log(PK)                                                       #
#   + epsilon                                                                #
##############################################################################
print("#" * 50)
print("Problem 2")
y = np.log(data[:, 0]).reshape(-1,1)  # 145 x 1 matrix; log(TC)
X = np.hstack(
    (np.ones((data.shape[0], 1)), np.log(data[:, [1, 2, 3, 4]]))
)  # 145 x 5 matrix; 1, log(Q), log(PL), log(PF), log(PK)

print("OLS estimator:")
for i, beta in enumerate(ols(y, X)[2]):
    print(f"beta_{i+1} = {beta}")

print("Standard errors:")
print(std_errors(y, X))

print("t-test:")
for i, (t, p) in enumerate(zip(*t_test(y, X))):
    print(f"beta_{i}: t = {t}, p = {p}")

print("R-squared:")
print(R_squared(y, X))

##############################################################################
# Problem 3                                                                  #
# log(TC / PF) =                                                             #
#   beta_1                                                                   #
#   + beta_2 * log(Q)                                                        #
#   + beta_3 * log(PL / PF)                                                  #
#   + beta_4 * log(PK / PF)                                                  #
#   + epsilon                                                                #
##############################################################################

print("#" * 50)
print("Problem 3")
y = np.log(data[:, 0] / data[:, 3])  # 145 x 1 matrix; log(TC / PF)
X = np.hstack(
    (
        np.ones((data.shape[0], 1)),
        np.log(data[:, 1]).reshape(-1, 1),
        np.log(data[:, 2] / data[:, 3]).reshape(-1, 1),
        np.log(data[:, 4] / data[:, 3]).reshape(-1, 1),
    )
)  # 145 x 5 matrix; 1, log(Q), log(PL / PF), log(PK / PF)

print("OLS estimator:")
for i, beta in enumerate(ols(y, X)[2]):
    print(f"beta_{i+1} = {beta}")

print("Standard errors:")
print(std_errors(y, X))

print("R-squared:")
print(R_squared(y, X))

##############################################################################
# Problem 4                                                                  #