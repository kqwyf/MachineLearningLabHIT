import numpy as np
import matplotlib.pyplot as plt
from logistic_model import *

N = 100
M = 10

dataset_file = "mammographic_masses.data"

def generate_covariance(size, independent=True):
    sigma = np.diag(np.abs(np.random.normal(size=size)) + 1)
    if not independent:
        _tmp_matrix = np.random.normal(size=(size,size))
        sigma = _tmp_matrix.T @ sigma @ _tmp_matrix
    return sigma

def generate_data(size, N, sigma=None, independent=True):
    mu = np.random.randint(11, size=size) - 5
    if sigma is None:
        sigma = generate_covariance(size, independent)
    print("average: " + str(mu))
    print("sigma:\n" + str(sigma))
    return np.random.multivariate_normal(mu, sigma, N)

def generate_dataset(size, N1, N2, independent=True, same_covariance=True):
    if same_covariance:
        sigma = generate_covariance(size, independent)
        print("class 1:")
        data1 = generate_data(size, N1, sigma=sigma, independent=independent)
        print("class 2:")
        data2 = generate_data(size, N2, sigma=sigma, independent=independent)
    else:
        print("class 1:")
        data1 = generate_data(size, N1, independent=independent)
        print("class 2:")
        data2 = generate_data(size, N2, independent=independent)
    return (np.vstack((data1, data2)), np.array([1]*N1 + [0]*N2))

def read_data(filename):
    x, y = [], []
    with open(filename, "r") as f:
        for line in f.readlines():
            line = line.strip().replace('?', '0').split(',')
            x += [np.array(line[:-1], dtype=np.int)]
            y += [int(line[-1])]
    N = len(x)
    N1 = int(0.7 * N)
    N2 = N - N1
    x, y = np.array(x), np.array(y)
    return len(x[0]), (x[:N1], y[:N1]), (x[N1:], y[N1:])

def accuracy(model, dataset):
    return np.sum(model.predict(dataset[0]) == dataset[1]) / len(dataset[1])

def plot(model, dataset):
    """
    Only for 2-dimension dataset.
    """
    x1 = np.min(dataset[0][:,0])
    x2 = np.max(dataset[0][:,0])
    y1 = -(model.w[0]*x1+model.w[2])/model.w[1]
    y2 = -(model.w[0]*x2+model.w[2])/model.w[1]

    plt.scatter(dataset[0][dataset[1]==1,0], dataset[0][dataset[1]==1,1], c='g')
    plt.scatter(dataset[0][dataset[1]==0,0], dataset[0][dataset[1]==0,1], c='r')
    plt.plot([x1,x2], [y1,y2])
    plt.show()

print("train set 2d:")
train_set_2d = generate_dataset(2, N, N, independent=True, same_covariance=False)
print("")
print("non-naive-bayes train set 1:")
train_set_1 = generate_dataset(2, N, N, independent=False, same_covariance=True)
print("")
print("non-naive-bayes train set 2:")
train_set_2 = generate_dataset(2, N, N, independent=True, same_covariance=False)
print("")
print("non-naive-bayes train set 3:")
train_set_3 = generate_dataset(2, N, N, independent=False, same_covariance=False)
print("")
print("train set:")
train_set = generate_dataset(M, N, N, independent=True, same_covariance=False)
print("")
real_M, real_set, real_validate_set = read_data(dataset_file)

##########################################

print("2-D generated dataset test (not regularized).")
model = logistic_model(2, l=0)
model.train_by_newton(train_set_2d, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set_2d))
plot(model, train_set_2d)
print("")

##########################################

print("10-D generated dataset test (not regularized).")
model = logistic_model(M, l=0)
model.train_by_newton(train_set, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set))
print("")

##########################################

print("2-D generated dataset test (regularized).")
model = logistic_model(2, l=1e-3)
model.train_by_newton(train_set_2d, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set_2d))
plot(model, train_set_2d)
print("")

##########################################

print("10-D generated dataset test (regularized).")
model = logistic_model(M, l=1e-3)
model.train_by_newton(train_set, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set))
print("")

##########################################

print("2-D generated non-naive-bayes dataset test (not regularized).")
print("The independent variables are correlated.")
model = logistic_model(2, l=0)
model.train_by_newton(train_set_1, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set_1))
plot(model, train_set_1)
print("")

##########################################

print("2-D generated non-naive-bayes dataset test (not regularized).")
print("The covariance matrices of 2 classes are different.")
model = logistic_model(2, l=0)
model.train_by_newton(train_set_2, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set_2))
plot(model, train_set_2)
print("")

##########################################

print("2-D generated non-naive-bayes dataset test (not regularized).")
print("The independent variables are correlated and the covariance matrices of 2 classes are different.")
model = logistic_model(2, l=0)
model.train_by_newton(train_set_3, num_step=10)
print("Accuracy: %.3f"%accuracy(model, train_set_3))
plot(model, train_set_3)
print("")

##########################################

print("Real dataset test (not regularized). dataset file: " + dataset_file)
model = logistic_model(real_M, l=0)
model.train_by_newton(real_set, num_step=10)
print("Train set accuracy: %.3f    Validate set accuracy: %.3f"%(accuracy(model, real_set), accuracy(model, real_validate_set)))
print("")

##########################################

print("Real dataset test (regularized). dataset file: " + dataset_file)
model = logistic_model(real_M, l=1e-3)
model.train_by_newton(real_set, num_step=10)
print("Train set accuracy: %.3f    Validate set accuracy: %.3f"%(accuracy(model, real_set), accuracy(model, real_validate_set)))
print("")
