# optimization.py

import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

### 
# Best fixed step size for this quadratic objective: 0.1.
# It stays below the stability bound 2 / lambda_max(H) = 2 / 16 = 0.125,
# while still converging to within 0.1 of the optimum quickly.
OPTIMAL_STEP_SIZE = 0.1
###

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='optimization.py')
    parser.add_argument('--func', type=str, default='QUAD', help='function to optimize (QUAD or NN)')
    parser.add_argument('--lr', type=float, default=1., help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    args = parser.parse_args()
    return args


def quadratic(x1, x2):
    """
    Quadratic function of two variables
    :param x1: first coordinate
    :param x2: second coordinate
    :return:
    """
    return (x1 - 1) ** 2 + 8 * (x2 - 1) ** 2


def quadratic_grad(x1, x2):
    """
    Differentiating the quadratic function f(x1, x2) = (x1 - 1)^2 + 8 * (x2 - 1)^2 with respect to x1 and x2 gives us the gradient:
    return: a one-dimensional numpy array containing two elements representing the gradient
    """
    return np.array([2.0 * (x1 - 1.0), 16.0 * (x2 - 1.0)], dtype=float)

def sgd_test_quadratic(args):
    xlist = np.linspace(-3.0, 3.0, 100)
    ylist = np.linspace(-3.0, 3.0, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = quadratic(X, Y)
    plt.figure()

    # Track the points visited here
    points_history = []
    curr_point = np.array([0., 0.])
    # prev_grad = None
    for iter in range(0, args.epochs):
        grad = quadratic_grad(curr_point[0], curr_point[1])
        if len(grad) != 2:
            raise Exception("Gradient must be a two-dimensional array (vector containing [df/dx1, df/dx2])")
        next_point = curr_point - args.lr * grad
        points_history.append(curr_point)
        distance_to_optimum = np.linalg.norm(next_point - np.array([1., 1.]))
        print("Iteration %d: distance_to_optimum=%0.6f" % (iter + 1, distance_to_optimum))
        curr_point = next_point
    points_history.append(curr_point)
    cp = plt.contourf(X, Y, Z)
    plt.colorbar(cp)
    plt.plot([p[0] for p in points_history], [p[1] for p in points_history], color='k', linestyle='-', linewidth=1, marker=".")
    plt.title('SGD on quadratic')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    exit()


if __name__ == '__main__':
    args = _parse_args()
    sgd_test_quadratic(args)
    #done
