import numpy as np


def ols(y, X):
    N, K = X.shape
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return (N, K, beta)


data = np.loadtxt("hw2/nerlove.asc")
y = np.log(data[:, 0])
X = data[:, [1, 2, 3, 4]]

N, K, beta = ols(y, X)
residual = y - X @ beta
degree_of_freedom = N - K
sigma_squared = (residual.T @ residual) / (N - K)
sigma = np.sqrt(sigma_squared)
variance_matrix = sigma_squared * np.linalg.inv(X.T @ X)
std_errors = np.sqrt(np.diag(variance_matrix))

if __name__ == "__main__":
    print(beta)
    print(std_errors)
