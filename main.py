
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from reservoir import Reservoir

random.seed(7)  # any integer
np.random.seed(7)



df = pd.read_csv('iris.csv', index_col=0)
df = pd.get_dummies(df)
df.head()

data_raw = df.to_numpy()
data_raw.shape

sample = lambda a, b: random.sample(range(a, b), 40)
train_idxs = sample(0, 50) + sample(50, 100) + sample(100, 150)
test_idxs = [idx for idx in range(150) if idx not in train_idxs]

# standardize data: divide each feature by maximum value (computed on training data only)
data = data_raw.copy()
data[:,:4] = data[:,:4] / data[train_idxs,:4].max(axis=0)

timesteps = np.arange(0, 50, 0.1)
u = np.array([
    np.vstack([np.sin(timesteps*2*np.pi*pt[i]) for i in range(4)]).T
    for pt in data[:,:4]
])
y = np.array([data[:,4:]] * len(timesteps)).swapaxes(0, 1)

fig, axs = plt.subplots(nrows=2, ncols=7, sharex=True, sharey=True, figsize=(20, 5), dpi=200)
for i in range(2):
    for j in range(7):
        axs[i][j].set_title(f'Input pattern #{i*7+j+1}')
        axs[i][j].plot(u[i*7+j,:100])
        axs[i][j].plot(y[i*7+j,:100])

u_train = u[train_idxs]
y_train = y[train_idxs]
u_test = u[test_idxs]
y_test = y[test_idxs]

# instantiate reservoir

reservoir = Reservoir(n_inputs=u_train.shape[2], n_neurons=20, rhow=1.25, leak_range=(0.1, 0.3))

print(f'{u_train.shape} -> {np.concatenate(u_train).shape}')

X_train = reservoir.forward(np.concatenate(u_train), collect_states=True)
X_train.shape

alpha = 1e-3
T_washout = 100

# drop states during the initial washout period of each time series
X_washout = X_train.reshape((u_train.shape[0], u_train.shape[1], -1))[:, T_washout:, :]
X = np.concatenate(X_washout)
Y = np.concatenate(y_train[:, T_washout:, :])
# state correlation matrix
R = X.T @ X
# state-output cross-correlation matrix
P = X.T @ Y
# ridge regression: state -> output
wout = np.linalg.inv((R + alpha * np.eye(X_train.shape[1]))) @ P

y_pred_train = X_train @ wout

X_test = reservoir.forward(np.concatenate(u_test), collect_states=True)
y_pred_test = X_test @ wout

def accuracy_elementwise(ypred, ytrue):
    # compute accuracy at each time step
    yp = ypred.reshape(-1, ytrue.shape[2]).argmax(axis=1)
    yt = ytrue.reshape(-1, ytrue.shape[2]).argmax(axis=1)
    return np.sum(yp == yt) / yp.shape[0]

print(f'Training accuracy: {accuracy_elementwise(y_pred_train, y_train):.2%} (per time step)')

def accuracy(ypred, ytrue):
    # compute accuracy by averaging the predicted class over the whole time series
    yp = ypred.reshape(ytrue.shape).mean(axis=1).argmax(axis=1)
    yt = ytrue[:,0,:].argmax(axis=1)
    return np.sum(yp == yt) / yp.shape[0]

#Reporting training data

print(f'Training accuracy: {accuracy(y_pred_train, y_train):.2%} (per time series)')

#Reporting test accuracy

print(f'Testing accuracy:  {accuracy(y_pred_test, y_test):.2%} (per time series)')

#print(f'Training accuracy: {accuracy(y_pred_train, y_train):.2%} (per time series)')
#print(f'Testing accuracy:  {accuracy(y_pred_test, y_test):.2%} (per time series)')