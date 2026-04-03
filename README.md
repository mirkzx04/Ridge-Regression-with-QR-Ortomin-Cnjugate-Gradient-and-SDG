# Ridge Regression Solution with QR Method, Ortomin, Conjugate Gradient and SGD

## Detailed Analysis of MATLAB Code

## 1. Problem Overview

### 1.1 The Ridge Regression Problem

The code solves the **Ridge Regression** problem, a regularization technique for linear regression. The mathematical problem is:

$$\min_{w} \|Xw - y\|_2^2 + \lambda \|w\|_2^2$$

where:
- $X \in \mathbb{R}^{n \times d}$ is the feature matrix
- $y \in \mathbb{R}^n$ is the target vector  
- $w \in \mathbb{R}^d$ are the weights to be estimated
- $\lambda > 0$ is the regularization parameter

### 1.2 Analytical Solution

The closed-form solution is:

$$w^* = (X^T X + \lambda I)^{-1} X^T y$$

The term $\lambda I$ makes the matrix always invertible and controls overfitting.

---

## 2. Main Function: `project()`

### 2.1 Configurable Parameters

```matlab
n      = 200;   % number of samples
d      = 50;    % number of features  
rng(1);         % fixed random seed for reproducibility
```

**Modifiable parameters:**
- `n`: Dataset size (number of observations)
- `d`: Number of features/variables
- `rng(seed)`: Seed for result reproducibility

### 2.2 Synthetic Data Generation

```matlab
X       = randn(n, d);
true_w  = randn(d, 1);
y       = X * true_w + 0.1 * randn(n, 1);
```

**Detailed explanation:**
- `X = randn(n, d)`: Generates feature matrix $X$ with elements from standard normal distribution $\mathcal{N}(0,1)$
- `true_w = randn(d, 1)`: Vector of "true" weights (ground truth) for comparison
- `y = X * true_w + 0.1 * randn(n, 1)`: Target with Gaussian noise ($\sigma = 0.1$)

**Modifiable parameters:**
- Noise level: `0.1` can be changed to increase/decrease noise
- Distribution: `randn` can be replaced with other distributions

### 2.3 Lambda Values Sweep

```matlab
lambdas   = logspace(-4, 3, 20);   % 20 values from 1e-4 to 1e3
times     = zeros(size(lambdas));
err_weights = zeros(size(lambdas));
err_rmse  = zeros(size(lambdas));
```

**Modifiable parameters:**
- Range of $\lambda$: `logspace(-4, 3, 20)` 
  - `-4`: minimum exponent ($10^{-4}$)
  - `3`: maximum exponent ($10^3$)
  - `20`: number of points in logarithmic range

**Evaluation loop:**
```matlab
for k = 1:numel(lambdas)
    [~, times(k), err_weights(k), err_rmse(k)] = ...
        ridge_reg_with_qr(X, y, lambdas(k), true_w);
end
```

For each $\lambda$ computes:
- `times(k)`: Solution time
- `err_weights(k)`: $\|w_{estimated} - w_{true}\|_2$
- `err_rmse(k)`: Root Mean Square Error

---

## 3. QR Method: `ridge_reg_with_qr()`

### 3.1 Mathematical Theory

The method transforms the problem into extended least squares form:

$$\min_w \left\| \begin{bmatrix} X \\ \sqrt{\lambda}I \end{bmatrix} w - \begin{bmatrix} y \\ 0 \end{bmatrix} \right\|_2^2$$

### 3.2 Detailed Implementation

```matlab
function [w_ridge, solution_time, err_weights, rmse] = ...
            ridge_reg_with_qr(X, y, lambda, true_w)
```

**Input:**
- `X`: Feature matrix $(n \times d)$
- `y`: Target vector $(n \times 1)$ 
- `lambda`: Regularization parameter
- `true_w`: True weights for comparison

**Extended matrix construction:**
```matlab
[~, d] = size(X);
Xtilde = [X; sqrt(lambda)*eye(d)];
ytilde = [y; zeros(d,1)];
```

- `Xtilde`: Matrix $(n+d) \times d$ that concatenates $X$ and $\sqrt{\lambda}I$
- `ytilde`: Vector $(n+d) \times 1$ that concatenates $y$ and zero vector

**QR decomposition and solution:**
```matlab
tic;
[Q, R]  = qr(Xtilde, 0);        % Compact QR
w_ridge = R \ (Q' * ytilde);   
solution_time = toc;
```

- `qr(Xtilde, 0)`: Compact QR decomposition ($Q$ has dimension $(n+d) \times d$)
- `R \ (Q' * ytilde)`: Solution of upper triangular system $Rw = Q^T \tilde{y}$

**Computational complexity:** $\mathcal{O}(nd^2)$

---

## 4. Conjugate Gradient Method: `ridge_reg_cg()`

### 4.1 Mathematical Theory

The Conjugate Gradient solves the linear system:
$$(X^T X + \lambda I)w = X^T y$$

It's an iterative method for symmetric positive definite systems.

### 4.2 CG Algorithm

```matlab
function [w_cg, it, res_hist] = ridge_reg_cg(X, y, lambda, tol, itmax)
```

**Modifiable parameters:**
- `tol = 1e-8`: Convergence tolerance
- `itmax = 1000`: Maximum number of iterations

**Initialization:**
```matlab
A = X.'*X + lambda*eye(size(X,2));
b = X.'*y;
w   = zeros(size(b));          % initial guess
r   = b - A*w;                 % initial residual
p   = r;                       % initial direction
rs_old = r' * r;
```

**Main loop:**
```matlab
while sqrt(rs_old) > tol && it < itmax
    it = it + 1;
    Ap = A*p;
    alpha = rs_old / (p' * Ap);    % Optimal step
    w  = w + alpha * p;            % Solution update
    r  = r - alpha * Ap;           % Residual update
    rs_new = r' * r;
    if sqrt(rs_new) < tol, break; end
    p = r + (rs_new / rs_old) * p; % New conjugate direction
    rs_old = rs_new;
end
```

**Key formulas:**
- Step: $\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$
- Update: $w_{k+1} = w_k + \alpha_k p_k$
- Conjugate direction: $p_{k+1} = r_{k+1} + \beta_k p_k$ with $\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$

---

## 5. ORTOMIN Method: `ridge_reg_ortomin()`

### 5.1 Mathematical Theory

ORTOMIN(k) is a generalization of CG that maintains orthogonality between residuals and previous search directions.

### 5.2 ORTOMIN(1) Implementation

```matlab
function [w_ortomin, it, res_hist] = ridge_reg_ortomin(X, y, lambda, tol, itmax)
```

**Differences from CG:**
```matlab
alpha = (r' * Ap) / (Ap' * Ap);     % Ortomin(k=1) step
beta  = (r_new' * Ap) / (Ap' * Ap); % Orthogonalization coefficient
```

**ORTOMIN(1) formulas:**
- Step: $\alpha_k = \frac{r_k^T A p_k}{p_k^T A^T A p_k}$
- Coefficient: $\beta_k = \frac{r_{k+1}^T A p_k}{p_k^T A^T A p_k}$

---

## 6. Stochastic Gradient Descent: `ridge_reg_sgd()`

### 6.1 Mathematical Theory

SGD minimizes the objective function using one sample at a time:

$$L_i(w) = \frac{1}{2}(x_i^T w - y_i)^2 + \frac{\lambda}{2}\|w\|_2^2$$

The gradient for sample $i$ is:
$$\nabla L_i(w) = (x_i^T w - y_i)x_i + \lambda w$$

### 6.2 Implementation

```matlab
function [w_sgd, history] = ridge_reg_sgd(X, y, lambda, opts)
```

**Configurable parameters in `opts`:**
- `opts.lr = 1e-3`: Learning rate
- `opts.epochs = 100`: Number of epochs
- `opts.shuffle = true`: Shuffle data at each epoch

**Main loop:**
```matlab
for e = 1:opts.epochs
    if opts.shuffle
        idx = randperm(n);
        X = X(idx,:); y = y(idx);
    end
    
    for i = 1:n
        xi = X(i,:)';                       
        yi = y(i);
        grad = (xi' * w_sgd - yi) * xi + lambda * w_sgd;  % Gradient
        w_sgd = w_sgd - opts.lr * grad;                   % Update
    end
end
```

**Update rule:**
$$w_{t+1} = w_t - \eta \nabla L_i(w_t)$$

where $\eta$ is the learning rate (`opts.lr`).

---

## 7. Method Comparison

### 7.1 Evaluation Metrics

The code compares methods on:

1. **Execution time**: Computational efficiency
2. **Weight error**: $\|w_{estimated} - w_{true}\|_2$
3. **RMSE**: $\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - x_i^T w)^2}$

### 7.2 Generated Plots

**First set of plots (lambda sweep):**
- Time vs $\lambda$
- Weight error vs $\lambda$ 
- RMSE vs $\lambda$

**Second set (convergence):**
- Residual vs iterations for iterative methods

**Third set (bar plot):**
- Final metric comparison

---

## 8. Main Modifiable Parameters

### 8.1 Dataset and Problem

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `n` | 200 | Number of samples |
| `d` | 50 | Number of features |
| Noise | 0.1 | Noise standard deviation |

### 8.2 Lambda Range

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `logspace(-4, 3, 20)` | $10^{-4}$ to $10^3$ | Regularization range |
| `lambda_test` | 1e-2 | Lambda for method comparison |

### 8.3 Iterative Methods

| Method | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| CG/ORTOMIN | `tol` | 1e-8 | Convergence tolerance |
| CG/ORTOMIN | `itmax` | 1000 | Maximum iterations |
| SGD | `opts.lr` | 1e-3 | Learning rate |
| SGD | `opts.epochs` | 200 | Number of epochs |
| SGD | `opts.shuffle` | true | Data shuffling |
