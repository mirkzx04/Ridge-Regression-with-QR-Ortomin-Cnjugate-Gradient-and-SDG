import numpy as np
import matplotlib.pyplot as plt
import time

def ridge_regression_qr(X, y, lam, true_w):
    """ Ridge regression using QR decomposition """

    # Expand X and y to include regularization term
    _, d = X.shape
    X_tilde = np.vstack([X, np.sqrt(lam) * np.eye(d)])
    y_tilde = np.vstack([y, np.zeros((d, 1))])
    
    # Measure time for QR decomposition
    start_time = time.time()
    Q, R = np.linalg.qr(X_tilde, 0)
    w_ridge = np.linalg.solve(R, Q.T @ y_tilde)
    end_time = time.time() - start_time

    # Metrics
    err_w = np.linalg.norm(w_ridge - true_w)
    y_pred = X @ w_ridge
    err_mse = np.sqrt(np.means(y_pred -y)**2)

    return w_ridge, err_w, err_mse, end_time

def ridge_regression_cg(X, y, lam, tol, max_iter):
    """ Ridge regression using Conjugate Gradient method """
    
    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y

    # Initial guess
    w = np.zeros((d, 1))
    r = b - A @ w
    p = r.copy()
    rsold = r.T @ r

    for i in range(max_iter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        w += alpha * p
        r -= alpha * Ap
        rsnew = r.T @ r
        
        if np.sqrt(rsnew) < tol:
            break
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return w

if __name__ == "__main__":
    # Base parameters
    n = 1000
    d = 200
    np.random.seed(42)

    # Generate random data
    X = np.random.randn(n, d)
    true_w = np.random.randn(d, 1)
    y = X @ true_w + 0.1 * np.random.randn(n, 1)

    # Sweep lambda
    lambdas = np.logspace(-4, 3, 20)
    lambda_sze = lambdas.size

    # Pre compute if matrics, row = Lambda, col = methods
    times = np.zeros((lambda_sze, 4)) # col1 = QR, col2 = CG, col3 = ORT, col4 = SGD
    err_w = np.zeros((lambda_sze, 4))  
    err_mse = np.zeros((lambda_sze, 4))

    lr = 1e-3  # Learning rate for SGD
    epochs = 1000  # Number of epochs for SGD
    shuffle = True  # Shuffle data for SGD

    for k in range(lambda_sze):
        lam = lambdas[k]
        
        # QR Method
        w_ridge, err_w[k, 0], err_mse[k, 0], times[k, 0] = ridge_regression_qr(X, y, lam, true_w)

        # CG Method
        start_time = time.time()
        w_cg = ridge_regression_cg(X, y, lam, 1e-8, 1000)

