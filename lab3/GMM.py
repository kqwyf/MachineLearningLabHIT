import numpy as np
from kmeans import kmeans

_EPS = 1e-6

# logarithm of the probability mass function of the gaussian disribution
def _log_normal(x, loc, scale_inv, log_det=None):
    D = np.shape(x)[-1]
    dx = x - loc # [d]
    if log_det == None:
        _, log_det = np.linalg.slogdet(scale_inv)
    return -(1/2) * ((dx@scale_inv@dx.T) + D*np.log(2*np.pi) - log_det) # 1

# logarithm of the probability distribution of the hidden variant
def _log_prob(X, loc, scale_inv, log_a):
    K = len(loc)
    log_det = [np.linalg.slogdet(s)[1] for s in scale_inv]
    if len(np.shape(X)) > 1:
        result = []
        for x in X:
            result += [[_log_normal(x, l, s, d) for (l,s,d) in zip(loc, scale_inv, log_det)]]
    else:
        result = [_log_normal(X, l, s, d) for (l,s,d) in zip(loc, scale_inv, log_det)]
    result = np.array(result)
    result += log_a
    return result

# logarithm of the likelihood function of the observed data
def _log_likelihood(X, loc, scale_inv, log_a):
    result = _log_prob(X, loc, scale_inv, log_a)
    result = np.mean(np.log(np.sum(np.exp(result), axis=1)))
    return result

class GMM:
    def __init__(self, K, eps=None):
        """
        Initialize the model.

        Parameters
        ----------
        K : int
            Number of classes.
        eps : float
            An small number used to control the iteration and normalize the scale.
        """
        self.log_a = None # weight of each sub-distribution
        self.loc = None # mean of each sub-distribution
        self.scale_inv = None # inversion of each covariance matrix
        self.K = K # number of classes
        self.EPS = _EPS if eps is None else eps
    def train_by_em(self, X, loc=None, num_step=None):
        """
        Train the model by EM algorithm.

        Parameters
        ----------
        X : array_like
            The train data.
        loc : array_like
            The initial means. If none, the initial means will be extracted by kmeans.
        num_step : int
            Number of iterations. If none, the train process won't stop as long as the likelihood
            change is greater than EPS.
        """
        K = self.K
        X = np.asarray(X)
        N = np.shape(X)[0] # number of datapoints
        D = np.shape(X)[-1] # number of dimensions of datapoints
        # initialize
        labels = kmeans(X, K)
        clus = [[] for i in range(K)]
        for (x, label) in zip(X, labels):
            clus[label] += [x]
        if loc is None:
            loc = np.array([np.mean(clus[i], axis=0) for i in range(K)])
        scale = np.array([(ci-mu).T @ (ci-mu) / len(ci) for (ci, mu) in zip(clus, loc)]) + self.EPS*np.eye(D)
        scale_inv = np.linalg.inv(scale)
        log_a = np.log([len(ci) for ci in clus]) - np.log(N)
        log_likelihood = _log_likelihood(X, loc, scale_inv, log_a)
        print("Training...")
        result = []
        step = 0
        while True:
            step += 1
            # calculate likelihood
            if num_step is None:
                print("\rtraining step: %d    likelihood: %f"%(step, log_likelihood), end='')
            else:
                print("\rtraining process: %.0f%%    likelihood: %f"%(step*100//num_step, log_likelihood), end='')
            result += [loc]
            # E step
            log_prob = _log_prob(X, loc, scale_inv, log_a) # calculate the log probability
            norm_prob = log_prob - np.log(np.sum(np.exp(log_prob), axis=1))[:,np.newaxis] # normalize
            prob = np.exp(norm_prob) # remove the log
            # M step
            ## update a
            Nk = np.sum(prob, axis=0)
            log_a = np.log(Nk) - np.log(N)
            ## update loc
            loc = (prob.T @ X) / Nk[:,np.newaxis]
            ## update scale and scale_inv
            scale = []
            for i in range(len(loc)):
                dX = X - loc[i]
                scale += [((dX.T * prob[:,i]) @ dX) / Nk[i]]
            scale += self.EPS*np.eye(D)
            scale_inv = np.linalg.inv(scale)
            # check if the iteration should be exited
            new_log_likelihood = _log_likelihood(X, loc, scale_inv, log_a)
            if num_step is None and abs(new_log_likelihood-log_likelihood)<self.EPS:
                break
            log_likelihood = new_log_likelihood
            if step >= 100: break
            if num_step is not None and step >= num_step:
                break
        print("\nTraining finished.    likelihood: %f"%(_log_likelihood(X, loc, scale_inv, log_a)))
        # update model
        self.log_a = log_a
        self.loc = loc
        self.scale_inv = scale_inv
        return result
    def predict(self, X):
        """
        Classify the given datapoints.

        Parameters
        ----------
        X : array_like
            The data to be classified.

        Returns
        -------
        result : array_like
            An 1-D array representing the labels of the data.
        """
        log_prob = _log_prob(X, self.loc, self.scale_inv, self.log_a)
        result = np.argmax(log_prob, axis=1)
        return result
