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
    start_time = time.perf_counter()
    Q, R = np.linalg.qr(X_tilde, mode="reduced")
    w_ridge = np.linalg.solve(R, Q.T @ y_tilde)
    end_time = time.perf_counter() - start_time

    # Metrics
    err_w = np.linalg.norm(w_ridge - true_w)
    y_pred = X @ w_ridge
    err_rmse = np.sqrt(np.mean((y_pred - y) ** 2))

    return w_ridge, end_time, err_w, err_rmse


def timed_qr(X, y, lam):
    """ Time only QR ridge regression solve """

    # Expand X and y to include regularization term
    _, d = X.shape
    X_tilde = np.vstack([X, np.sqrt(lam) * np.eye(d)])
    y_tilde = np.vstack([y, np.zeros((d, 1))])

    # Measure time for QR decomposition
    start_time = time.perf_counter()
    Q, R = np.linalg.qr(X_tilde, mode="reduced")
    _ = np.linalg.solve(R, Q.T @ y_tilde)
    end_time = time.perf_counter() - start_time

    return end_time


def ridge_regression_cg(X, y, lam, tol=1e-8, max_iter=1000):
    """ Ridge regression using Conjugate Gradient method """

    n, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y

    # Initial guess
    w = np.zeros((d, 1))
    r = b - A @ w
    p = r.copy()
    rsold = float(r.T @ r)

    for _ in range(max_iter):
        Ap = A @ p
        denom = float(p.T @ Ap)

        if abs(denom) < 1e-30:
            break

        alpha = rsold / denom
        w = w + alpha * p
        r = r - alpha * Ap
        rsnew = float(r.T @ r)

        if np.sqrt(rsnew) < tol:
            break

        p = r + (rsnew / rsold) * p
        rsold = rsnew

    return w


def ridge_regression_ortomin(X, y, lam, tol=1e-8, max_iter=1000):
    """ Ridge regression using ORTOMIN method """

    # Solve (X'X + lambda I)w = X'y with Ortomin(k=1)
    _, d = X.shape
    A = X.T @ X + lam * np.eye(d)
    b = X.T @ y

    # Initial guess
    w = np.zeros((d, 1))
    r = b - A @ w
    p = r.copy()

    for _ in range(max_iter):
        if np.linalg.norm(r) <= tol:
            break

        Ap = A @ p
        denom = float(Ap.T @ Ap)

        if abs(denom) < 1e-30:
            break

        alpha = float(r.T @ Ap) / denom
        w = w + alpha * p
        r_new = r - alpha * Ap

        beta = float(r_new.T @ Ap) / denom
        p = r_new + beta * p
        r = r_new

    return w


def ridge_regression_sgd(X, y, lam, opts=None):
    """ Ridge regression using SGD """

    # opts.lr        : learning rate
    # opts.epochs    : epochs
    # opts.shuffle   : shuffle data each epoch
    # returns:
    #   w_sgd   : estimated weights

    if opts is None:
        opts = {}

    lr = opts.get("lr", 1e-3)
    epochs = opts.get("epochs", 100)
    shuffle = opts.get("shuffle", True)

    n, d = X.shape
    w_sgd = np.zeros((d, 1))  # init to zero

    for _ in range(epochs):
        if shuffle:
            idx = np.random.permutation(n)
            X_epoch = X[idx, :]
            y_epoch = y[idx, :]
        else:
            X_epoch = X
            y_epoch = y

        for i in range(n):
            xi = X_epoch[i, :].reshape(d, 1)  # sample (d x 1)
            yi = y_epoch[i, 0]

            # Gradient of Ridge cost for single sample
            grad = (float(xi.T @ w_sgd) - yi) * xi + lam * w_sgd
            w_sgd = w_sgd - lr * grad

    return w_sgd


if __name__ == "__main__":
    # Base parameters
    n = 500
    d = 200
    seed = 1
    np.random.seed(seed)

    # Generate random data
    X = np.random.randn(n, d)
    true_w = np.random.randn(d, 1)
    y = X @ true_w + 0.1 * np.random.randn(n, 1)

    # Sweep lambda
    lambdas = np.logspace(-4, 3, 20)
    lambda_size = lambdas.size

    # Pre compute metrics, row = lambda, col = methods
    times = np.zeros((lambda_size, 4))     # col1 = QR, col2 = CG, col3 = ORT, col4 = SGD
    err_w = np.zeros((lambda_size, 4))
    err_rmse = np.zeros((lambda_size, 4))

    opts = {
        "lr": 1e-3,        # Learning rate for SGD
        "epochs": 200,    # Number of epochs for SGD
        "shuffle": True   # Shuffle data for SGD
    }

    for k in range(lambda_size):
        lam = lambdas[k]

        # QR Method
        w_qr, times[k, 0], err_w[k, 0], err_rmse[k, 0] = ridge_regression_qr(
            X, y, lam, true_w
        )

        # CG Method
        start_time = time.perf_counter()
        w_cg = ridge_regression_cg(X, y, lam, tol=1e-8, max_iter=1000)
        times[k, 1] = time.perf_counter() - start_time
        err_w[k, 1] = np.linalg.norm(w_cg - true_w)
        err_rmse[k, 1] = np.sqrt(np.mean((X @ w_cg - y) ** 2))

        # ORTOMIN Method
        start_time = time.perf_counter()
        w_ort = ridge_regression_ortomin(X, y, lam, tol=1e-8, max_iter=1000)
        times[k, 2] = time.perf_counter() - start_time
        err_w[k, 2] = np.linalg.norm(w_ort - true_w)
        err_rmse[k, 2] = np.sqrt(np.mean((X @ w_ort - y) ** 2))

        # SGD Method
        start_time = time.perf_counter()
        w_sgd = ridge_regression_sgd(X, y, lam, opts)
        times[k, 3] = time.perf_counter() - start_time
        err_w[k, 3] = np.linalg.norm(w_sgd - true_w)
        err_rmse[k, 3] = np.sqrt(np.mean((X @ w_sgd - y) ** 2))

    # --------- Compact figure with 3 metrics ----------
    labels = ["QR", "CG", "ORT", "SGD"]
    styles = ["-o", "--d", "-.s", ":^"]  # QR CG ORT SGD
    draw_order = [0, 1, 3, 2]             # QR, CG, SGD, finally ORT

    # micro-jitter to separate overlapping curves
    jitter = np.array([1.0, 1.0 + 5e-4, 1.0 + 1e-3, 1.0 + 1.5e-3])

    plt.figure("Sweep lambda - all metrics", figsize=(9, 7))

    # Time
    plt.subplot(3, 1, 1)
    plt.grid(True)
    for j in draw_order:
        linewidth = 2.0 if j in [1, 2] else 1.4
        plt.semilogx(
            lambdas * jitter[j],
            times[:, j],
            styles[j],
            linewidth=linewidth,
            label=labels[j]
        )
    plt.ylabel("Time [s]")
    plt.title("Solve time")
    plt.legend()

    # Weight error
    plt.subplot(3, 1, 2)
    plt.grid(True)
    for j in draw_order:
        linewidth = 2.0 if j in [1, 2] else 1.4
        plt.semilogx(
            lambdas * jitter[j],
            err_w[:, j],
            styles[j],
            linewidth=linewidth,
            label=labels[j]
        )
    plt.ylabel(r"$\|w-w_{true}\|_2$")
    plt.yscale("log")
    plt.title("Estimated weights error vs true weights")
    plt.legend()

    # RMSE
    plt.subplot(3, 1, 3)
    plt.grid(True)
    for j in draw_order:
        linewidth = 2.0 if j in [1, 2] else 1.4
        plt.semilogx(
            lambdas * jitter[j],
            err_rmse[:, j],
            styles[j],
            linewidth=linewidth,
            label=labels[j]
        )
    plt.xlabel(r"$\lambda$")
    plt.ylabel("RMSE")
    plt.yscale("log")
    plt.title("Prediction RMSE")
    plt.legend()

    plt.tight_layout()

    # Time vs lambda on extended range
    lambdas_big = np.logspace(-20, 3, 24)
    tbig = np.zeros((lambdas_big.size, 4))

    for k in range(lambdas_big.size):
        lam = lambdas_big[k]

        tbig[k, 0] = timed_qr(X, y, lam)

        start_time = time.perf_counter()
        _ = ridge_regression_cg(X, y, lam, tol=1e-8, max_iter=2000)
        tbig[k, 1] = time.perf_counter() - start_time

        start_time = time.perf_counter()
        _ = ridge_regression_ortomin(X, y, lam, tol=1e-8, max_iter=2000)
        tbig[k, 2] = time.perf_counter() - start_time

        start_time = time.perf_counter()
        _ = ridge_regression_sgd(X, y, lam, opts)
        tbig[k, 3] = time.perf_counter() - start_time

    plt.figure("Time vs lambda extended range", figsize=(8, 4.5))
    plt.grid(True)
    for j in range(len(labels)):
        plt.semilogx(
            lambdas_big,
            tbig[:, j],
            "-o",
            linewidth=1.4,
            label=labels[j]
        )
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Time [s]")
    plt.title(r"Execution time as a function of $\lambda$ (n=500, d=200)")
    plt.legend()
    plt.tight_layout()

    # Time vs d with fixed n for four lambda values
    lam_list = np.array([1e-4, 1e-2, 1.0, 1e2])
    dims2 = np.array([20, 50, 100, 200, 400, 800])
    T_d = np.zeros((dims2.size, lam_list.size, 4))  # dims x lambda x method

    for di in range(dims2.size):
        dcur = dims2[di]
        Xd = np.random.randn(n, dcur)
        wtrue_d = np.random.randn(dcur, 1)
        yd = Xd @ wtrue_d + 0.1 * np.random.randn(n, 1)

        for li in range(lam_list.size):
            lam = lam_list[li]

            # QR
            T_d[di, li, 0] = timed_qr(Xd, yd, lam)

            # CG
            start_time = time.perf_counter()
            _ = ridge_regression_cg(Xd, yd, lam, tol=1e-8, max_iter=2000)
            T_d[di, li, 1] = time.perf_counter() - start_time

            # ORT
            start_time = time.perf_counter()
            _ = ridge_regression_ortomin(Xd, yd, lam, tol=1e-8, max_iter=2000)
            T_d[di, li, 2] = time.perf_counter() - start_time

            # SGD
            start_time = time.perf_counter()
            _ = ridge_regression_sgd(Xd, yd, lam, opts)
            T_d[di, li, 3] = time.perf_counter() - start_time

    # Plot one graph for each method
    for meth in range(4):
        plt.figure(f"Time vs d - {labels[meth]}")
        plt.grid(True)

        for li in range(lam_list.size):
            plt.plot(
                dims2,
                T_d[:, li, meth],
                "-o",
                linewidth=1.4,
                label=rf"$\lambda={lam_list[li]:.0e}$"
            )

        plt.xlabel("d")
        plt.ylabel("Time [s]")
        plt.title(f"{labels[meth]}: time vs d (n={n})")
        plt.legend()
        plt.tight_layout()

    # Execution time as a function of dimension d, fixed n and lambda
    lambda0 = 1e-2
    dims = np.array([20, 50, 100, 200, 400, 800])
    nd = dims.size
    time_d = np.zeros((nd, 4))  # QR CG ORT SGD

    for jd in range(nd):
        d_cur = dims[jd]
        Xd = np.random.randn(n, d_cur)
        w_true_d = np.random.randn(d_cur, 1)
        yd = Xd @ w_true_d + 0.1 * np.random.randn(n, 1)

        # QR
        _, t_qr, _, _ = ridge_regression_qr(Xd, yd, lambda0, w_true_d)
        time_d[jd, 0] = t_qr

        # CG
        start_time = time.perf_counter()
        _ = ridge_regression_cg(Xd, yd, lambda0, tol=1e-8, max_iter=2000)
        time_d[jd, 1] = time.perf_counter() - start_time

        # ORT
        start_time = time.perf_counter()
        _ = ridge_regression_ortomin(Xd, yd, lambda0, tol=1e-8, max_iter=2000)
        time_d[jd, 2] = time.perf_counter() - start_time

        # SGD
        start_time = time.perf_counter()
        _ = ridge_regression_sgd(Xd, yd, lambda0, opts)
        time_d[jd, 3] = time.perf_counter() - start_time

    # Plot times vs dimension
    plt.figure("Time vs dimension d", figsize=(7, 4.5))
    plt.grid(True)

    for j in range(len(labels)):
        plt.plot(
            dims,
            time_d[:, j],
            "-o",
            linewidth=1.4,
            label=labels[j]
        )

    plt.xlabel("d, number of features")
    plt.ylabel("Time [s]")
    plt.title(rf"Execution time (n={n}, $\lambda={lambda0:.0e}$)")
    plt.legend()
    plt.tight_layout()

    plt.show()