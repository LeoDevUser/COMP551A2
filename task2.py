import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg') #Set GUI backend for plots


np.random.seed(42)#set seed for reproducibility
#get our sample data

def get_bases(D,x):
    '''retuns numpy.ndarray of D Gaussian bases'''
    gaussian = lambda x, mu, sigma: np.exp(-((x-mu)/sigma)**2)
    mu = []
    for d in range(D):
        mu.append(np.min(x) + (np.max(x)-np.min(x))/(D-1) * d)
    mu = np.array(mu)
    phi = gaussian(x[:,None], mu[None,:],1)
    return phi

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

#Cost Function for prediction
def sse(y_predict, y_validation):
    sse = 0
    for i in range(len(y_predict)):
        sse += (y_predict[i] - y_validation[i])**2

    return sse

epsilon = np.random.normal(0,1,100)
f_clean = lambda x : np.log(x+1) * np.cos(x) + np.sin(2*x)
f_noise = lambda x : np.log(x+1) * np.cos(x) + np.sin(2*x) + epsilon
x_uniform = np.linspace(0,10,100)
true_y = f_clean(x_uniform)

#to hold the data for the 10 samplings
x = []
y_clean = []
y_noise = []
X_train = []
X_test = []
Y_train = []
Y_test = []

#to hold sse results
sses_train = [[],[],[],[],[],[],[],[],[],[]] #array to hold the sse of train set of each model
sses_validation = [[],[],[],[],[],[],[],[],[],[]] #array to hold the sse of validation set of each model

#sample data 10 times
for t in range(10):
    x_local = np.random.rand(100) * 10

    y_clean_local = f_clean(x_local)
    y_noise_local = f_noise(x_local)

    #Split Data 80/20
    X_train_local = x_local[:80]
    X_test_local = x_local[80:]
    Y_train_local = y_noise_local[:80]
    Y_test_local = y_noise_local[80:]
    #sort for better visualization
    #sort test splits
    sorted_train = np.argsort(X_train_local)
    sorted_test = np.argsort(X_test_local)
    X_train.append(X_train_local[sorted_train])
    X_test.append(X_test_local[sorted_test])
    Y_train.append(Y_train_local[sorted_train])
    Y_test.append(Y_test_local[sorted_test])

    sorted_indices = np.argsort(x_local)
    x.append(x_local[sorted_indices])
    y_clean.append(y_clean_local[sorted_indices])
    y_noise.append(y_noise_local[sorted_indices])

#get average x
average_x  = []
for t in range(len(X_train[0])):
    xi = []
    for q in range(len(X_train)):
        xi.append(X_train[q][t])
    average_x.append(np.mean(xi))

#start plotting 10 models
fig = plt.figure(figsize=(20, 8))
#plot 0 bases
fits = []
phi_train = np.empty((len(X_train[0]), 0))
phi_test = np.empty((len(X_test[0]), 0))
plt.subplot(2, 5, 1)
plt.plot(x_uniform, true_y, '-b')
for t in range(10):
    plt.title('0 bases')
    #Fit and Predict empty feature matrix since zero bases
    model = LinearRegression().fit(phi_train,Y_train[t])
    yh_train = model.predict(phi_train)
    fits.append(yh_train)
    plt.plot(X_train[t], yh_train, '-g', alpha=.5)
    #plot the true function
    #compute sse for the model
    sses_train[0].append(sse(yh_train, Y_train[t]))
    yh_validation = model.predict(phi_test)
    sses_validation[0].append(sse(yh_validation, Y_test[t]))

#plot average
plt.plot(average_x, np.mean(fits,axis=0), '-r')

#plot the 9 remaining models
for t in range(9):
    plt.subplot(2, 5, t+2)
    plt.title(f'{(t+1) * 5} bases')
    fits = []
    for q in range(10):
        phi = get_bases((t+1)*5,X_train[q])
        phi_test = get_bases((t+1)*5,X_test[q])
        model = LinearRegression().fit(phi,Y_train[q])
        yh_train = model.predict(phi)
        fits.append(yh_train)
        plt.plot(X_train[q], yh_train, '-g', alpha=.5)
        #plot the true function
        #compute sse for the model
        sses_train[t+1].append(sse(yh_train, Y_train[q]))
        yh_validation = model.predict(phi_test)
        sses_validation[t+1].append(sse(yh_validation, Y_test[q]))
    #plot ground truth
    plt.plot(x_uniform, true_y, '-b')

    #plot average fit
    plt.plot(average_x,np.mean(fits, axis=0), '-r')


plt.savefig('task2.png')
plt.show()

#make table with sse results
basis_counts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('tight')
ax.axis('off')

#Prepare table data
table_data = [['Bases', 'Train SSE', 'Validation SSE']]
for i, d in enumerate(basis_counts):
    table_data.append([str(d), f'{np.mean(sses_train[i]):.4f}', f'{np.mean(sses_validation[i]):.4f}'])

#Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)  # Scale height of cells
plt.tight_layout()
plt.savefig('sseTask2.png',dpi=150,bbox_inches='tight')
plt.show()
