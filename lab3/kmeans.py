import numpy as np

def kmeans(x, k, means=None):
    # initialize the means and labels
    if means is None:
        means = x[np.random.randint(0, len(x), size=k)]
    labels = np.argmin([np.sum((means - p)**2, 1) for p in x], 1)
    # update means and labels iteratively
    while True:
        # separate data into classes
        xi = [[] for _ in range(k)]
        for i in range(len(labels)):
            xi[labels[i]] += [x[i]]
        # remove classes with no datapoints
        xi = list(filter(lambda x: len(x)>0, xi))
        # update means and labels
        means = np.array([np.average(xi[i] if len(xi[i])>0 else [(0,0)], 0) for i in range(k)])
        new_labels = np.argmin([np.sum((means - p)**2, 1) for p in x], 1)
        # check if the result changed
        result = new_labels == labels
        if np.alltrue(result):
            break
        labels = new_labels
    return labels
