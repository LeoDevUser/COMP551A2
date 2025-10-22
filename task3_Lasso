# Task 3 — Lasso (L1) 
import numpy as np
import matplotlib.pyplot as plt


# True function and data

def f_true(x):
    x = np.asarray(x)
    return np.log(x + 1) * np.cos(x) + np.sin(2 * x)

def sample_dataset(N=100, xmin=0.0, xmax=10.0, noise_std=1.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = rng.uniform(xmin, xmax, size=N)
    y = f_true(x) + rng.normal(0.0, noise_std, size=N)
    return x, y


# Gaussian basis (D=45)

def gaussian_centers(D=45, xmin=0.0, xmax=10.0):
    return np.linspace(xmin, xmax, D)

def gaussian_basis(x, centers, sigma=1.0, include_bias=True):
    x = np.asarray(x).reshape(-1, 1)
    mu = np.asarray(centers).reshape(1, -1)
    Phi = np.exp(-((x - mu) ** 2) / (sigma ** 2))
    if include_bias:
        Phi = np.column_stack([Phi, np.ones(Phi.shape[0])])
    return Phi


# Cross-validation 

def cross_validate(n, n_folds=10, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    n_val = n // n_folds
    inds = rng.permutation(n)

    for f in range(n_folds):
        start = f * n_val
        stop = (f + 1) * n_val if f < n_folds - 1 else n
        val_inds = list(inds[start:stop])
        tr_inds = list(np.concatenate([inds[:start], inds[stop:]]))
        yield tr_inds, val_inds


# Error

def mse(y_true, y_pred):
    return np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)

def _soft_threshold(z, gamma):
    return np.sign(z) * np.maximum(np.abs(z) - gamma, 0.0)


# Lasso 

class LassoRegressor:
 
    def __init__(self, lambda_=0.0, penalize_bias=False, max_iter=5000, tol=1e-6):
        self.lambda_ = float(lambda_)
        self.penalize_bias = penalize_bias
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.w = None

    def fit(self, Phi, y):
        Phi = np.asarray(Phi, float)
        y = np.asarray(y, float).reshape(-1)
        N, D = Phi.shape
        w = np.zeros(D)
        bias_idx = D - 1  
        aj = (Phi ** 2).sum(axis=0) / N

        for _ in range(self.max_iter):
            w_old = w.copy()
            
            for j in range(D):
                r = y - (Phi @ w) + w[j] * Phi[:, j]
                cj = (Phi[:, j] * r).sum() / N

                if (not self.penalize_bias) and (j == bias_idx):
                    w[j] = cj / (aj[j] + 1e-12)
                else:
                    w[j] = _soft_threshold(cj, self.lambda_) / (aj[j] + 1e-12)

            if np.linalg.norm(w - w_old, ord=np.inf) < self.tol:
                break

        self.w = w
        return self

    def predict(self, Phi):
        return np.asarray(Phi, float) @ self.w


# Configuration

rng = np.random.default_rng(1337)
D = 45
sigma = 1.0
N_total = 100
n_folds = 10
lambda_grid = np.logspace(-3, 1, 10)
penalize_bias = False


# Data

x, y = sample_dataset(N=N_total, rng=rng)
centers = gaussian_centers(D, 0.0, 10.0)
Phi = gaussian_basis(x, centers, sigma, include_bias=True)


# 10-fold cross-validation across lambda

train_mse = []
val_mse = []

for lam in lambda_grid:
    fold_train = []
    fold_val = []
    for tr_idx, va_idx in cross_validate(len(x), n_folds=n_folds, rng=rng):
        Phi_tr, y_tr = Phi[tr_idx], y[tr_idx]
        Phi_va, y_va = Phi[va_idx], y[va_idx]

        model = LassoRegressor(lambda_=lam, penalize_bias=penalize_bias).fit(Phi_tr, y_tr)
        yhat_tr = model.predict(Phi_tr)
        yhat_va = model.predict(Phi_va)

        fold_train.append(mse(y_tr, yhat_tr))
        fold_val.append(mse(y_va, yhat_va))

    train_mse.append(np.mean(fold_train))
    val_mse.append(np.mean(fold_val))

best_idx = int(np.argmin(val_mse))
best_lambda = float(lambda_grid[best_idx])
print(f"[Lasso] Best lambda (by CV): {best_lambda:.5g}")


# Training/Validation MSE vs lambda

plt.figure(figsize=(7,5))
plt.plot(lambda_grid, train_mse, marker='o', label='Training MSE')
plt.plot(lambda_grid, val_mse, marker='o', label='Validation MSE')
plt.xscale('log')
plt.xlabel('lambda (log scale)')
plt.ylabel('MSE')
plt.title('Lasso Regression: Training/Validation MSE vs lambda')
plt.legend()
plt.tight_layout()
plt.show()


# Bias–Variance Decomposition

num_datasets = 50
N_per_dataset = 20
x_test = np.linspace(0, 10, 200)
Phi_test = gaussian_basis(x_test, centers, sigma, include_bias=True)
f_test = f_true(x_test)
noise_var = 1.0

predictions = np.zeros((num_datasets, len(lambda_grid), len(x_test)))

for d in range(num_datasets):
    xd, yd = sample_dataset(N=N_per_dataset, rng=rng)
    Phid = gaussian_basis(xd, centers, sigma, include_bias=True)

    for i, lam in enumerate(lambda_grid):
        model = LassoRegressor(lambda_=lam, penalize_bias=penalize_bias).fit(Phid, yd)
        predictions[d, i, :] = model.predict(Phi_test)

mean_pred = predictions.mean(axis=0)
var_pred  = predictions.var(axis=0)
bias2     = (mean_pred - f_test[None, :])**2

bias2_curve = bias2.mean(axis=1)
var_curve   = var_pred.mean(axis=1)
bv_curve    = bias2_curve + var_curve
bvn_curve   = bv_curve + noise_var

# Empirical test MSE
test_mse = []
for i in range(len(lambda_grid)):
    mse_i = 0.0
    for d in range(num_datasets):
        y_test_noisy = f_test + rng.normal(0.0, np.sqrt(noise_var), size=len(x_test))
        mse_i += mse(y_test_noisy, predictions[d, i, :])
    test_mse.append(mse_i / num_datasets)
test_mse = np.array(test_mse)


# Bias–Variance vs lambda

plt.figure(figsize=(8,5))
plt.plot(lambda_grid, bias2_curve, marker='o', label='Bias^2')
plt.plot(lambda_grid, var_curve, marker='o', label='Variance')
plt.plot(lambda_grid, bv_curve, marker='o', label='Bias^2 + Variance')
plt.plot(lambda_grid, bvn_curve, marker='o', label='Bias^2 + Variance + Noise(σ^2=1)')
plt.plot(lambda_grid, test_mse, marker='o', label='Empirical Test MSE')
plt.xscale('log')
plt.xlabel('lambda (log scale)')
plt.ylabel('Error')
plt.title('Lasso: Bias–Variance Decomposition')
plt.legend()
plt.tight_layout()
plt.show()






# Lambda=0 vs Lasso Regression

# Fitting full-data models
lasso_unreg = LassoRegressor(lambda_=0.0, penalize_bias=penalize_bias,
                             max_iter=20000, tol=1e-10).fit(Phi, y)
lasso_best  = LassoRegressor(lambda_=best_lambda, penalize_bias=penalize_bias).fit(Phi, y)

plt.figure(figsize=(8,5))

# Grid for true function and Lasso
x_grid   = np.linspace(0, 10, 400)
Phi_grid = gaussian_basis(x_grid, centers, sigma, include_bias=True)
y_true_grid  = f_true(x_grid)
y_lasso_grid = lasso_best.predict(Phi_grid)    

# Unregularized predictions 
order         = np.argsort(x)
x_train_line  = x[order]
y_unreg_line  = lasso_unreg.predict(Phi)[order]


y_for_limits = np.concatenate([y, y_lasso_grid, y_true_grid])
q1, q2 = np.percentile(y_for_limits, [1, 99])
yr = q2 - q1
ymin, ymax = q1 - 0.15*yr, q2 + 0.15*yr

# Plot
plt.scatter(x, y, s=20, alpha=0.6, label='Noisy data', zorder=3)
plt.plot(x_grid, y_true_grid, lw=2, alpha=0.8, label='True function', zorder=2)
plt.plot(x_grid, y_lasso_grid, lw=2, label='Lasso', zorder=2)
plt.plot(x_train_line, y_unreg_line, '-o', lw=1.6, ms=3.5, alpha=0.95,
         label='Unregularized', zorder=1)

plt.title('True function vs. Lasso vs. Unregularized')
plt.xlabel('x'); plt.ylabel('y')
plt.xlim(0, 10)
plt.ylim(ymin, ymax)
plt.legend()
plt.tight_layout()
plt.show()
