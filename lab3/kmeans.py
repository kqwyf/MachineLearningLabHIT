import numpy as np

def kmeans(x, k, centers=None):
    k = min(len(x), k)
    if centers is None:
        centers = x[np.random.randint(0, len(x), size=k)]
    labels = np.argmin([np.sum((centers - p)**2, 1) for p in x], 1)
    while True:
        xi = [[] for _ in range(k)]
        for i in range(len(labels)):
            xi[labels[i]] += [x[i]]
        centers = np.array([np.average(xi[i] if len(xi[i])>0 else [(0,0)], 0) for i in range(k)])
        new_labels = np.argmin([np.sum((centers - p)**2, 1) for p in x], 1)
        result = new_labels == labels
        if np.alltrue(result):
            break
        else:
            print("rest: %d"%(len(result)-np.sum(result)))
        labels = new_labels
    return labels
