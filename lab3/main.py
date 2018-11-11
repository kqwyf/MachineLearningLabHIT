import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture.gaussian_mixture import *
import csv
from kmeans import kmeans
from GMM import GMM

K = 3
n = 100
colors = ['r', 'g', 'b']
tcolors = ['yellow', 'black', 'gray']

def generate_points_near(x, y, n, scale=1.0):
    return np.random.multivariate_normal((x,y), scale*np.eye(2), n)

centers = np.random.randint(-10, 10, (K,2))
scales = np.random.randint(1, 10, K)
points = np.vstack([generate_points_near(center[0], center[1], n) for center in centers])
k_labels = kmeans(points, K)
"""import pickle
f=open("data.in","rb")
centers, scales, points, k_labels = pickle.load(f)
f.close()"""
labels = np.hstack([[i]*n for i in range(K)])

plt.subplot(2,4,1)
for i in range(K):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])

plt.subplot(2,4,2)
labels=k_labels
mid=[]
for i in range(K):
    mid += [np.average(points[labels==i], 0)]
mid = np.array(mid)
for i in range(K):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])
plt.scatter(mid[:,0], mid[:,1], c='black')

def draw_circle(plt, x, y, r):
    import math
    tmp = np.linspace(0, 2*math.pi, 100)
    x = r*np.cos(tmp) + x
    y = r*np.sin(tmp) + y
    plt.plot(x, y)

loc = np.array([np.average(points[labels==i], 0) for i in range(K)]) # [k,d]
model = GMM(K)
for step in range(5):
    plt.subplot(2,4,3+step)
    result = model.train_by_em(points, num_step=step+1, loc=loc)
    labels = model.predict(points)
    for i in range(K):
        plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])
    result = np.array(result)
    for i in range(K):
        plt.plot(result[:,i,0], result[:,i,1], c=tcolors[i])
plt.subplot(2,4,8)
result = model.train_by_em(points, loc=loc)
labels = model.predict(points)
for i in range(K):
    plt.scatter(points[labels==i, 0], points[labels==i, 1], c=colors[i])
result = np.array(result)
for i in range(K):
    plt.plot(result[:,i,0], result[:,i,1], c=tcolors[i])
    draw_circle(plt, model.loc[i,0], model.loc[i,1], 1/np.linalg.det(model.scale_inv[i]))
plt.show()

print("real data:")
#k = max(int(np.sqrt(data.shape[0]//2)), 1)
k = 3
model = GMM(k)
with open("seeds.txt", "r") as f:
    #data = np.array(list(csv.reader(f)), dtype=np.float)
    data = []
    label = []
    for line in f:
        words = line.split()
        data += [[float(i) for i in words[:-1]]]
        label += [int(words[-1])]
    data = np.array(data)
    label = np.array(label)
model.train_by_em(data)
labels = model.predict(data)
"""
model = GaussianMixture(10)
model.fit(data)
labels = model.predict(data)
"""
tmp = np.array([[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]])
print("accuracy: %.2f"%(max([np.sum(label==tmp[i][labels]) for i in range(6)])/len(data)))
for i in range(k):
    plt.scatter(data[labels==i, 2], data[labels==i, 1], c=colors[i])
plt.show()
