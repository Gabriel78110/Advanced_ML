from soft_svm import SoftSVM
from data import *
import matplotlib.pyplot as plt
import random

train, test = get_data()
x_range = np.linspace(-20,10,10000)

for C in [0.1, 1, 50]:
    svm = SoftSVM(C=C)
    svm.train(train[0], train[1], test[0],test[1],n_iterations=10000, learning_rate=10**(-5))
    '''row_sample = random.sample(range(0,train[0].shape[0]), int(0.1*train[0].shape[0]))
    x = np.take(train[0], row_sample, axis=0)
    y = np.take(train[1], row_sample, axis=0)
    SV_1, NSV_1, SV_0, NSV_0 = [], [], [], []
    for i in range(y.shape[0]):
        if y[i]==1 and y[i]*np.dot(svm.w,x[i,:]) <=1:
            SV_1.append(i)

        elif y[i]==1 and y[i]*np.dot(svm.w,x[i,:]) > 1:
            NSV_1.append(i)

        elif y[i]==-1 and y[i]*np.dot(svm.w,x[i,:]) <=1:
            SV_0.append(i)

        elif y[i]==-1 and y[i]*np.dot(svm.w,x[i,:] > 1):
            NSV_0.append(i)

    plt.figure()
    if len(SV_1)!=0:
        SV_1 = np.array(SV_1)
        x_sv1 = np.take(x, SV_1, axis=0)
        plt.scatter(x_sv1[:,1],x_sv1[:,2], c='red', label='1 and support vector')
    if len(NSV_1)!=0:
        NSV_1 = np.array(NSV_1)
        x_nsv1 = np.take(x, NSV_1, axis=0)
        plt.scatter(x_nsv1[:,1],x_nsv1[:,2], c='blue', label='1 and non-support vector')
    if len(SV_0)!=0:
        SV_0 = np.array(SV_0)
        x_sv0 = np.take(x, SV_0, axis=0)
        plt.scatter(x_sv0[:,1],x_sv0[:,2], c='yellow', label='-1 and support vector')
    if len(NSV_0)!=0:
        NSV_0 = np.array(NSV_0)
        x_nsv0 = np.take(x, NSV_0, axis=0)
        plt.scatter(x_nsv0[:,1],x_nsv0[:,2], c='green', label='-1 and non-support vector')

    plt.plot(x_range,(-svm.w[0]-svm.w[1]*x_range)/svm.w[2], label='separator line')
    plt.title('C = {}'.format(C))
    plt.legend()
    plt.savefig(f'C_{C}.png')
    plt.show()'''
