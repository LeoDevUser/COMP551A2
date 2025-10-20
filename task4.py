import numpy as np
#%matplotlib inline
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import warnings
warnings.filterwarnings('ignore')
import itertools

class GradientDescent:
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8, record_history=False):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.record_history = record_history
        self.epsilon = epsilon
        if record_history:
            self.w_history = []
            
    def run(self, gradient_fn, x, y, w):
        grad = np.inf
        t = 1
        while np.linalg.norm(grad) > self.epsilon and t < self.max_iters:
            grad = gradient_fn(x, y, w)
            w = w - self.learning_rate * grad
            if self.record_history:
                self.w_history.append(w)
            t += 1
        return w
    
class LinearRegression:
    def __init__(self, add_bias=True, add_l2= False, add_l1 =False, reg_coeff=0):
        self.add_bias = add_bias
        self.add_l2 = add_l2
        self.add_l1 = add_l1
        self.reg_coeff = reg_coeff
        pass
            
    def fit(self, x, y, optimizer):
        if x.ndim == 1:
            x = x[:, None]
        if self.add_bias:
            N = x.shape[0]
            x = np.column_stack([x,np.ones(N)])
        N,D = x.shape
        def gradient(x, y, w):
            yh =  x @ w 
            N, D = x.shape
            grad = .5*np.dot(yh - y, x)/N
            if self.add_l1:                                 #add L1 regularization
                grad += self.reg_coeff*np.sign(w)
            if self.add_l2:                                 #add L2 regularization
                grad += self.reg_coeff* w
            return grad
        w0 = np.zeros(D)
        self.w = optimizer.run(gradient, x, y, w0)
        return self
    
    def predict(self, x):
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w
        return yh
    
def plot_contour(f, x1bound, x2bound, resolution, ax):
    x1range = np.linspace(x1bound[0], x1bound[1], resolution)
    x2range = np.linspace(x2bound[0], x2bound[1], resolution)
    xg, yg = np.meshgrid(x1range, x2range)
    zg = np.zeros_like(xg)
    for i,j in itertools.product(range(resolution), range(resolution)):
        zg[i,j] = f([xg[i,j], yg[i,j]])
    ax.contour(xg, yg, zg, 100)
    return ax

#generate synthetic data:
#30 data points
N = 30
#30 x values uniformly distributed between 1 and 10
x = np.linspace(0, 10, N)
#y = -3x + 8 +2E with E normally distributed with mean 0 and var 1
y = -3*x + 8 + 2*np.random.randn(N)

#linear cost function
cost = lambda w: .5*np.mean((w[0] + w[1]*x - y)**2)
#penalty functions for L1 and L2 regularization
l2_penalty = lambda w: np.dot(w,w)/2
l1_penalty = lambda w: np.sum(np.abs(w))

#regularization coefficients to plot
reg_list_l2 = [0, 1, 10]
reg_list_l1 = [0, 10, 20]


#cost contours for L2
fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 5))
plot_contour(cost, [-10,10], [-5,5], 50, axes[0])
axes[0].set_title(r'cost function $J(w)$')
plot_contour(l2_penalty, [-10,10], [-5,5], 50, axes[1])
axes[1].set_title(r'L2 reg. $||w||_2^2$')
plt.show()

#cost contours for L1
fig, axes = plt.subplots(ncols=2, nrows=1, constrained_layout=True, figsize=(10, 5))
plot_contour(cost, [-10,10], [-5,5], 50, axes[0])
axes[0].set_title(r'cost function $J(w)$')
plot_contour(l1_penalty, [-10,10], [-5,5], 50, axes[1])
axes[1].set_title(r'L1 reg. $||w||_1$')
plt.show()

#cost functions with regularization
cost_plus_l2 = lambda w, reg: cost(w) + reg*l2_penalty(w)
cost_plus_l1 = lambda w, reg: cost(w) + reg*l1_penalty(w)

#plot of cost function vs. regularization for L2 (contours of loss w trajectory of optimizer)
fig, axes = plt.subplots(ncols=len(reg_list_l2), nrows=1, constrained_layout=True, figsize=(15, 5))
for i, reg_coef in enumerate(reg_list_l2):
    optimizer = GradientDescent(learning_rate=.01, max_iters=50, record_history=True)
    model = LinearRegression(optimizer, add_l2=True, reg_coeff=reg_coef)
    model.fit(x,y, optimizer)
    current_cost = lambda w: cost_plus_l2(w, reg_coef)
    plot_contour(current_cost, [-20,20], [-5,5], 50, axes[i])
    w_hist = np.vstack(optimizer.w_history)# T x 2
    axes[i].plot(w_hist[:,1], w_hist[:,0], '.r', alpha=.8)
    axes[i].plot(w_hist[:,1], w_hist[:,0], '-r', alpha=.3)
    axes[i].set_xlabel(r'$w_0$')
    axes[i].set_ylabel(r'$w_1$')
    axes[i].set_title(r'L2 reg. cost $J(w) + ||w||_2^2$'+'\n'+ f' lambda = {reg_coef}')
    axes[i].set_xlim([-10,10])
    axes[i].set_ylim([-5,5])
plt.show()

#plot of cost function vs. regularization for L1 (contours of loss w trajectory of optimizer)
fig, axes = plt.subplots(ncols=len(reg_list_l1), nrows=1, constrained_layout=True, figsize=(15, 5))
for i, reg_coef in enumerate(reg_list_l1):
    optimizer = GradientDescent(learning_rate=.01, max_iters=50, record_history=True)
    model = LinearRegression(optimizer, add_l1=True, reg_coeff=reg_coef)
    model.fit(x,y, optimizer)
    current_cost = lambda w: cost_plus_l1(w, reg_coef)
    plot_contour(current_cost, [-20,20], [-5,5], 50, axes[i])
    w_hist = np.vstack(optimizer.w_history)# T x 2
    axes[i].plot(w_hist[:,1], w_hist[:,0], '.r', alpha=.8)
    axes[i].plot(w_hist[:,1], w_hist[:,0], '-r', alpha=.3)
    axes[i].set_xlabel(r'$w_0$')
    axes[i].set_ylabel(r'$w_1$')
    axes[i].set_title(r'L1 reg. cost $J(w) + ||w||_1$'+'\n'+ f' lambda = {reg_coef}')
    axes[i].set_xlim([-10,10])
    axes[i].set_ylim([-5,5])
plt.show()
