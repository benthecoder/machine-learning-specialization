import numpy as np
import math, copy
import matplotlib.pyplot as plt

def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    # for every data point, sum up the cost
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2

    # average the cost
    total_cost = 1 / (2 * m) * cost

    return total_cost

def compute_gradient(x, y, w, b):

    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    # for every data point, sum up the partial differentials
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    # average the partial differentials
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in) # avoid changing the original w
    
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # compute gradient and update parameters
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # update w and b
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        if i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # print cost at intervals 10 times
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e}")
            print(f"dj_dw: {dj_dw : 0.3e}, dj_db: {dj_db : 0.3e}")
            print(f"w: {w:0.3e}, b: {b:0.5e}\n")

    return w, b, J_history, p_history

if __name__ == "__main__":
    x_train = np.array([1.0, 2.0])   #features
    y_train = np.array([300.0, 500.0])   #target value

    # initialize params
    w_init = 0
    b_init = 0

    # gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2

    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

    print(f"(w, b) found by gradient descent: ({w_final:0.3e}, {b_final:0.5e})")


    # plot cost versus iteration  
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
    ax1.plot(J_hist[:100])
    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
    ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
    ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
    plt.show()