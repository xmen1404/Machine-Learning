# Fitting for n-rd polynomial
# Calculating the gradient decent by taking derivative of matrices
# Work well with learning_rate = 0.000008 and 10000000 updating changes (loss ~ 0.3)
# Still a bit time-consuming

import numpy as np
import matplotlib.pyplot as plt

def prepare_x(_x, _cvariable): 
    num = len(_x)
    _x = _x.reshape((num, 1))

    x_tmp = np.hstack((np.ones((num, 1)), _x))
    tmp = _x
    for i in range(_cvariable - 2): 
        tmp = tmp * _x
        x_tmp = np.hstack((x_tmp, tmp))
    return x_tmp

def prepare_data(_x, _y, _cvariable): 
    # process values of x
    _x = prepare_x(_x, _cvariable)
    row, col = _x.shape
    _y = _y.reshape((row, 1))
    return _x, _y

def loss_cal(_theta, _x, _y): 
    return (1 / len(_x)) * (_x @ _theta - _y).T @ (_x @ _theta - _y) 

def fitting_gd(_x, _y): 
    alpha = 0.000008
    print(x.shape, y.shape)
    # theta = np.array([[-7.4015], [10.3], [-2.4], [0.1577]])
    theta = np.zeros((len(_x[0]), 1))

    best_theta = theta
    best_loss = loss_cal(theta, _x, _y)[0][0]


    for i in range(10000000): 
        gd_val = (2 / len(_x)) * _x.T @ (_x @ theta - _y)
        theta = theta - alpha * gd_val
        # print(gd_val)

        tmp_loss = loss_cal(theta, _x, _y)[0][0]

        if tmp_loss < best_loss: 
            best_loss = tmp_loss
            best_theta = theta

    print(best_loss)
    return best_theta

x = np.array([1, 2, 4, 5, 6, 9]) 
y = np.array([1, 4, 6, 3, 1, 4])
plt.plot(x, y, marker = 'o', label = 'data points', ls = '')

x, y = prepare_data(x, y, 4)
print(x)

theta = fitting_gd(x, y)
x_predict = np.linspace(0, 10, 100)
x_tmp = prepare_x(x_predict, 4)
y_predict = (x_tmp @ theta).reshape(-1)

plt.plot(x_predict, y_predict, label = 'prediction', ls = '-')
plt.show()


# plt.plot(x, y, marker = 'o', label = 'data points', ls = '')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.plot(x, poly(x), ls = '-', label = 'fitted')
# plt.show()





