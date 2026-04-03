# Cracking Ridge Regression: From QR Methods to SGD
This guide breaks down a MATLAB implementation of **Ridge Regression**, exploring how we can solve for weights using everything from classic matrix factorizations to iterative modern solvers.

---

## 1. The Big Picture: What are we solving?

At its heart, Ridge Regression is about finding the best balance. We want to fit our data well, but we also want to keep our weights $w$ small to prevent the model from "overreacting" to noise (overfitting).

### The Mathematical Goal
We minimize the combined cost of error and model complexity:
$$\min_{w} \|Xw - y\|_2^2 + \lambda \|w\|_2^2$$

The solution is elegant and stable:
$$w^* = (X^T X + \lambda I)^{-1} X^T y$$
The $\lambda I$ term is our "safety net"—it ensures the matrix is always invertible and keeps our predictions grounded.

---

## 2. Setting the Stage: The `project()` Function

Before solving, we need a playground. The `project()` function sets up a synthetic universe where we actually know the "truth," allowing us to test how well our algorithms recover it.

### The Setup
* **Dimensions:** We start with $n=200$ samples and $d=50$ features.
* **Ground Truth:** We generate a `true_w` and create target values $y$ by adding a little Gaussian noise ($0.1$). 
* **The Lambda Sweep:** Since we don't know the perfect $\lambda$ upfront, the code tests 20 different values across a logarithmic scale from $10^{-4}$ to $10^3$.

---

## 3. Method 1: The QR Decomposition
**The "Gold Standard" for Stability**

Rather than calculating $(X^TX)^{-1}$ directly (which can be numerically grumpy), we use a clever trick. We stack the regularization directly into the data matrix:

$$\text{Solve for } w: \begin{bmatrix} X \\ \sqrt{\lambda}I \end{bmatrix} w \approx \begin{bmatrix} y \\ 0 \end{bmatrix}$$

**Why QR?** By decomposing this extended matrix into $Q$ (orthogonal) and $R$ (upper triangular), we turn a complex inversion into a simple back-substitution. It’s robust, precise, and handles the $\mathcal{O}(nd^2)$ complexity like a champ.

---

## 4. Method 2: Conjugate Gradient (CG)
**The Efficiency Specialist**

CG is an iterative powerhouse. Instead of solving everything in one giant leap, it takes strategic steps. It’s specifically designed for symmetric, positive-definite systems like ours.

* **The Strategy:** It searches for the solution by moving in "conjugate" directions, ensuring that each step doesn't undo the progress of the previous one.
* **Key Controls:** You can tune the `tol` (how much error you’re willing to tolerate) and `itmax` (when to give up and stop).

---

## 5. Method 3: ORTOMIN(1)
**The Orthogonal Optimizer**

Think of ORTOMIN as a cousin to CG. While CG focuses on error residuals, ORTOMIN(1) works to maintain a strict orthogonality between the current residual and the previous search direction. It’s a slightly different flavor of iteration that can be more stable in specific numerical environments.

---

## 6. Method 4: Stochastic Gradient Descent (SGD)
**The "One at a Time" Approach**

SGD is the engine behind modern Deep Learning. Unlike the other methods that look at the whole dataset (Batch), SGD looks at one single row $x_i$ at a time.

* **The Logic:** It calculates a "mini-gradient" for one sample, takes a small step (the `learning rate`), and repeats.
* **Shuffle for Success:** We shuffle the data every epoch to make sure the model doesn't get stuck in a repetitive loop. 
* **The Update Rule:**
    $$w_{t+1} = w_t - \eta [(x_i^T w - y_i)x_i + \lambda w]$$

---

## 7. How do we pick a winner?

The code compares these four contenders using three main "scorecards":

1.  **Speed (Time):** How long did it take to reach the answer?
2.  **Weight Accuracy:** How close did we get to the `true_w`?
3.  **RMSE:** How well do our predicted $y$ values match the actual targets?



---

## 8. Your Tuning Knobs

Want to experiment? Here are the most impactful variables you can change in the code:

| If you want to change... | Look for... | Impact |
| :--- | :--- | :--- |
| **Problem Complexity** | `n` and `d` | Larger numbers make the problem harder and slower. |
| **Regularization Strength** | `lambdas` | Higher $\lambda$ prevents overfitting but might "dampen" the truth. |
| **SGD Speed** | `opts.lr` | If the learning rate is too high, SGD "explodes"; too low, and it crawls. |
| **Precision** | `tol` | Tightening the tolerance makes CG/ORTOMIN more accurate but slower. |

---
