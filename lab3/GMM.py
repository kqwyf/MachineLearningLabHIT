import numpy as np
import scipy.linalg
from kmeans import kmeans
import math

_EPS = 1e-6

def _log_normal(x, loc, scale_inv, log_det=None):
    D = np.shape(x)[-1]
    dx = x - loc # [d]
    if log_det == None:
        _, log_det = np.linalg.slogdet(scale_inv)
    return -(1/2) * ((dx@scale_inv@dx.T) + D*np.log(2*np.pi) - log_det) # 1
    #return -(1/2) * (np.sum(np.square((dx@scale_inv))) + D*np.log(2*np.pi) - log_det) # 1

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

def _log_likelihood(X, loc, scale_inv, log_a):
    result = _log_prob(X, loc, scale_inv, log_a)
    result = np.mean(np.log(np.sum(np.exp(result), axis=1)))
    return result
"""
    _, log_det = np.linalg.slogdet(scale_inv)
    log_det = np.sum(log_det)
    log_prob = np.array([_log_normal(x, loc[i], scale_inv[i], log_det) for i in range(k)]).T
    weighted_log_prob = log_prob + np.log(a)
    log_prob_norm = np.log(np.sum(np.exp(weighted_log_prob), axis=1))
    return np.mean(log_prob_norm)
"""

class GMM:
    def __init__(self, K, eps=None):
        self.log_a = None
        self.loc = None
        self.scale_inv = None
        self.K = K
        self.EPS = _EPS if eps is None else eps
    def train_by_em(self, X, loc=None, num_step=None):
        K = self.K
        X = np.asarray(X)
        N = np.shape(X)[0]
        D = np.shape(X)[-1]
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
            log_prob = _log_prob(X, loc, scale_inv, log_a)
            norm_prob = log_prob - np.log(np.sum(np.exp(log_prob), axis=1))[:,np.newaxis]
            prob = np.exp(norm_prob)
            # M step
            Nk = np.sum(prob, axis=0)
            log_a = np.log(Nk) - np.log(N)
            loc = (prob.T @ X) / Nk[:,np.newaxis]
            scale = []
            for i in range(len(loc)):
                dX = X - loc[i]
                scale += [((dX.T * prob[:,i]) @ dX) / Nk[i]]
            scale += self.EPS*np.eye(D)
            if np.any(np.linalg.det(scale) == 0):
                print("\n\n\nSINGULAR!!!\n\n")
                break
            scale_inv = np.linalg.inv(scale)
            new_log_likelihood = _log_likelihood(X, loc, scale_inv, log_a)
            if num_step is None and math.fabs(new_log_likelihood-log_likelihood)<self.EPS:
                break
            log_likelihood = new_log_likelihood
            if step >= 100: break
            if num_step is not None and step >= num_step:
                break
        print("\nTraining finished.    likelihood: %f"%(_log_likelihood(X, loc, scale_inv, log_a)))
        self.log_a = log_a
        self.loc = loc
        self.scale_inv = scale_inv
        return result
    def predict(self, X):
        log_prob = _log_prob(X, self.loc, self.scale_inv, self.log_a)
        result = np.argmax(log_prob, axis=1)
        return result
