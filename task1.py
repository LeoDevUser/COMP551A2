import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') #Set GUI backend for plots


np.random.seed(42)#set seed for reproducibility
#get our sample data
x = np.random.rand(100) * 10
epsilon = np.random.normal(0,1,100)

#function definitions
f_clean = lambda x : np.log(x+1) * np.cos(x) + np.sin(2*x)
f_noise = lambda x : np.log(x+1) * np.cos(x) + np.sin(2*x) + epsilon
y_clean = f_clean(x)
y_noise = f_noise(x)
x_uniform = np.linspace(0,10,100)
true_y = f_clean(x_uniform)

#Split Data 80/20 split as usual
#TODO
X_train = x[:80]
X_test = x[80:]
Y_train = y_noise[:80]
Y_test = y_noise[80:]
#sort for better visualization
#sort test splits
sorted_train = np.argsort(X_train)
sorted_test = np.argsort(X_test)
X_train = X_train[sorted_train]
X_test = X_test[sorted_test]
Y_train = Y_train[sorted_train]
Y_test = Y_test[sorted_test]

sorted_indices = np.argsort(x)
x = x[sorted_indices]
y_clean = y_clean[sorted_indices]
y_noise = y_noise[sorted_indices]

#1.1
plt.title('Clean and Noisy Data plot')
plt.plot(x, y_clean, '.', label='Clean')
plt.plot(x, y_noise, '.', label='Noisy')
#plot the true function
plt.plot(x_uniform, true_y, alpha=.5, label='True Function')
plt.legend()
plt.savefig('cleannoisy.png')
plt.show()

#Gaussian Bases
#1.2
def get_bases(D,x):
    '''retuns numpy.ndarray of D Gaussian bases'''
    gaussian = lambda x, mu, sigma: np.exp(-((x-mu)/sigma)**2)
    mu = []
    for d in range(D):
        mu.append(np.min(x) + (np.max(x)-np.min(x))/(D-1) * d)
    mu = np.array(mu)
    phi = gaussian(x[:,None], mu[None,:],1)
    return phi

D=45
#plot the 45 gaussian bases
for d in range(D):
    plt.plot(x_uniform, get_bases(D,x_uniform)[:,d],'-')
plt.xlabel('x')
plt.title('Gaussian Bases')
plt.savefig('bases.png')
plt.show()

class LinearRegression:
    def __init__(self, add_bias=True):
        self.add_bias = add_bias
        pass

    def fit(self,x,y):
        if x.ndim == 1:
            x = x[:,None]
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        self.w = np.linalg.lstsq(x,y)[0]
        return self

    def predict(self,x):
        N = x.shape[0]
        if self.add_bias:
            x = np.column_stack([x,np.ones(N)])
        yh = x@self.w
        return yh

#1.3 and 1.4
#Cost Function for prediction
def sse(y_predict, y_validation):
    sse = 0
    for i in range(len(y_predict)):
        sse += (y_predict[i] - y_validation[i])**2

    return sse

#to hold sse results
sses_train = [] #array to hold the sse of train set of each model
sses_validation = [] #array to hold the sse of validation set of each model

fig = plt.figure(figsize=(20, 8))
plt.subplot(2, 5, 1)
#Fit and Predict empty feature matrix since zero bases
#create empty feature matrix (0 features)
phi_train = np.empty((len(X_train), 0))  # 80 x 0 matrix
phi_test = np.empty((len(X_test), 0))  # 80 x 0 matrix
model = LinearRegression().fit(phi_train,Y_train)
yh_train = model.predict(phi_train)
plt.title('0 bases')
plt.plot(X_train, yh_train, '-', label='Fit')
plt.plot(X_train, Y_train, '.', label='Noisy')
#plot the true function
plt.plot(x_uniform, true_y, alpha=.5, label='True Function')
plt.legend()
#compute sse for the model
sses_train.append(sse(yh_train, Y_train))
yh_validation = model.predict(phi_test)
sses_validation.append(sse(yh_validation, Y_test))

#plot the remaining 9 models
for t in range(9):
    plt.subplot(2, 5, t+2)
    phi = get_bases((t+1)*5,X_train)
    phi_test = get_bases((t+1)*5,X_test)
    model = LinearRegression().fit(phi,Y_train)
    yh_train = model.predict(phi)
    plt.title(f'{(t+1)*5} bases')
    plt.plot(X_train, yh_train, '-', label='Fit')
    plt.plot(X_train, Y_train, '.', label='Noisy')
    #plot the true function
    plt.plot(x_uniform, true_y, alpha=.5, label='True Function')
    plt.legend()
    #compute sse for the model
    sses_train.append(sse(yh_train, Y_train))
    yh_validation = model.predict(phi_test)
    sses_validation.append(sse(yh_validation, Y_test))

plt.savefig('models.png')
plt.show()

#make table with sse results
basis_counts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

#Prepare table data
table_data = [['Bases', 'Train SSE', 'Validation SSE']]
for i, d in enumerate(basis_counts):
    table_data.append([str(d), f'{sses_train[i]:.4f}', f'{sses_validation[i]:.4f}'])

#Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)  # Scale height of cells
plt.tight_layout()
plt.savefig('sse.png',dpi=150,bbox_inches='tight')
plt.show()
