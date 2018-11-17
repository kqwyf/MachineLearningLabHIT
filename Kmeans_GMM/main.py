import numpy as np
import matplotlib.pyplot as plt
import csv
from kmeans import kmeans
from GMM import GMM

K = 3 # number of classes
n = 100 # data size of each class
colors = ['r', 'g', 'b'] # color of each class

# generate points of a specific gaussian distribution
def generate_points_near(x, y, n, scale=np.eye(2)):
    return np.random.multivariate_normal((x,y), scale, n)

# draw a circle in pyplot
def draw_circle(plt, x, y, r):
    import math
    tmp = np.linspace(0, 2*math.pi, 100)
    x = r*np.cos(tmp) + x
    y = r*np.sin(tmp) + y
    plt.plot(x, y)

#################################

print("generated data:")

# generate data
means = np.random.randint(-10, 10, (K,2))
scales = [np.diag(np.random.randint(1, 5, 2)) for _ in range(K)]
points = np.vstack([generate_points_near(mean[0], mean[1], n, scale) for mean,scale in zip(means, scales)])
labels = np.hstack([[i]*n for i in range(K)])

# show the generated data
plt.subplot(1,3,1)
for i in range(K):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])

# show the kmeans result
plt.subplot(1,3,2)
labels = kmeans(points, K)
means = np.array([np.average(points[labels==i], 0) for i in range(K)])
for i in range(K):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])
plt.scatter(means[:,0], means[:,1], c='black') # mean points found by kmeans

# show the EM result
plt.subplot(1,3,3)
model = GMM(K)
loc = np.array([np.average(points[labels==i], 0) for i in range(K)]) # initialize by kmeans
result = model.train_by_em(points, loc=loc)
labels = model.predict(points)
for i in range(K):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])
    draw_circle(plt, model.loc[i,0], model.loc[i,1], np.sqrt(1/np.linalg.det(model.scale_inv[i])))
plt.show()
plt.plot(range(len(result)), result) # likelihood
plt.show()

#################################

print("real data:")

# read data
with open("seeds.txt", "r") as f:
    data = []
    label = []
    for line in f:
        words = line.split()
        data += [[float(i) for i in words[:-1]]]
        label += [int(words[-1])]
    data = np.array(data)
    label = np.array(label)

# train model
k = 3
model = GMM(k)
result = model.train_by_em(data)
labels = model.predict(data)
tmp = np.array([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])
print("accuracy: %.2f"%(max([np.sum(label==tmp[i][labels]) for i in range(6)])/len(data)))

# show result
for i in range(k):
    plt.scatter(data[labels==i, 2], data[labels==i, 1], c=colors[i])
plt.show()
plt.plot(range(len(result)), result) # likelihood
plt.show()
