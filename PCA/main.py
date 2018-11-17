import numpy as np
import matplotlib.pyplot as plt
import pickle
import PCA
import read_data

K = 16
N = 100
EPS = 1e-6

def generate_data(n):
    x = np.zeros(3)
    while np.linalg.norm(x) < EPS:
        x = np.random.normal(scale=100, size=3)
    x /= np.linalg.norm(x)
    y = np.zeros(3)
    while np.linalg.norm(y) < EPS:
        y = np.random.normal(scale=100, size=3)
        y = y - np.dot(y, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    space = np.vstack((x,y,z))
    points = np.array([np.random.normal(scale=0.1,size=n), np.random.normal(scale=20,size=n), np.random.normal(scale=20,size=n)]).T
    return points @ space

def read_mnist_data():
    train_images = read_data.decode_idx3_ubyte("train-images-idx3-ubyte", False, None)
    train_labels = read_data.decode_idx1_ubyte("train-labels-idx1-ubyte")
    test_images = read_data.decode_idx3_ubyte("t10k-images-idx3-ubyte", False, None)
    test_labels = read_data.decode_idx1_ubyte("t10k-labels-idx1-ubyte")
    return (train_images, train_labels), (test_images, test_labels)

def snr(real_data, fake_data):
    real_data = np.reshape(real_data, -1)
    fake_data = np.reshape(fake_data, -1)
    return 10 * np.log10(sum(fake_data**2) / sum((real_data-fake_data)**2))

data = generate_data(N)
pca, fake_data = PCA.extract_pca(data, 2)
# real data - first view
plt.subplot(131)
plt.xlim(np.min(data), np.max(data))
plt.ylim(np.min(data), np.max(data))
plt.scatter(data[:,0], data[:,1])
# real data - second view
plt.subplot(132)
plt.xlim(np.min(data), np.max(data))
plt.ylim(np.min(data), np.max(data))
plt.scatter(data[:,2], data[:,1])
# fake data
plt.subplot(133)
plt.xlim(np.min(data), np.max(data))
plt.ylim(np.min(data), np.max(data))
plt.scatter(fake_data[:,0], fake_data[:,1])
plt.show()

# MNIST images test
with open("mnist.data", "rb") as f:
    (train_images, train_labels), (test_images, test_labels) = pickle.load(f)

images = np.reshape(train_images, (len(train_images),-1))
pca, data = PCA.extract_pca(images, K);
images = PCA.recover_from_pca(pca, data)
images = np.reshape(images, (-1, 28, 28))

t_images = np.reshape(test_images, (len(test_images), -1))
t_data = t_images @ pca.T
t_images = PCA.recover_from_pca(pca, t_data)
t_images = np.reshape(t_images, (-1, 28, 28))

snr1 = np.average([snr(train_images[i], images[i]) for i in range(len(images))])
snr2 = np.average([snr(test_images[i], t_images[i]) for i in range(len(t_images))])
print("K = %d"%K)
print("SNR for train set: %.3f"%snr1)
print("SNR for test set: %.3f"%snr2)

for j in range(4):
    i = np.random.randint(0, len(images))
    plt.subplot(2, 4, j*2+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.subplot(2, 4, j*2+2)
    plt.imshow(images[i], cmap='gray')
plt.show()

for j in range(10):
    plt.subplot(4, 4, j+1)
    plt.imshow(np.reshape(pca[j], (28,28)), cmap='gray')
plt.show()
