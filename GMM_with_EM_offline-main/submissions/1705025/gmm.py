import numpy as np
from scipy.special import logsumexp




class GMM:
    def __init__(self, data, dim, k, max_iter=100, bias = 1e-5):
        
        self.data = data
        self.max_iter = max_iter
        self.dim = dim
        self.k = k
        self.n = data.shape[0]
        self.bias = bias

        #print all the parameters
        print("dim: ", self.dim, "k: ", self.k, "n: ", self.n)

        self.mu_all = []
        self.sigma_all = []


    def initialize(self):
        #initialize mean matrix
        
        self.mu = np.random.rand(self.k, self.dim)
        self.mu_all.append(self.mu)
        
        #initialize covariance matrix
        self.sigma = np.array([np.identity(self.dim) for i in range(self.k)])
        self.sigma_all.append(self.sigma)
        
        #initialize pi matrix
        self.pi = np.ones(self.k) / self.k

        


    def expectation(self):
        log_likelihood = 0
        gamma = np.zeros((self.n, self.k))

        for i in range(self.n):
            x = self.data[i]
            log_probs = []

            for j in range(self.k):
                mean = self.mu[j]
                cov = self.sigma[j]
                weight = self.pi[j]

                #calculate the log probability
                if weight == 0:
                    weight+=self.bias
                log_weight = np.log(weight)
                log_gaussian_prob = self.getLogGaussianProb(x, mean, cov)
                log_prob = log_weight + log_gaussian_prob
                log_probs.append(log_prob)

            log_likelihood += logsumexp(log_probs)
            gamma[i] = np.exp(log_probs - logsumexp(log_probs))

        return log_likelihood, gamma


    def maximization(self, gamma):
        gamma_sum = np.sum(gamma, axis=0)


        eps = 1e-9

        if np.any(gamma_sum[:, np.newaxis] < eps):
            self.mu = np.dot(gamma.T, self.data) / (gamma_sum[:, np.newaxis] + eps)
        else:
            self.mu = np.dot(gamma.T, self.data) / (gamma_sum[:, np.newaxis])

        for i in range(self.k):
            diff = self.data - self.mu[i]

            if np.any(gamma_sum[i] < eps):
                self.sigma[i] = np.dot(gamma[:, i] * diff.T, diff) / (gamma_sum[i] + eps)
            else:
                self.sigma[i] = np.dot(gamma[:, i] * diff.T, diff) / gamma_sum[i]

        self.pi = gamma_sum / self.n

    
    def EM(self):
        self.initialize()
        last_log_likelihood = 0


        for i in range(self.max_iter):
           
            log_likelihood, gamma = self.expectation()
            self.maximization(gamma)

            if self.checkConvergence(last_log_likelihood, log_likelihood):
                print("Converged at iteration: ", i)
                break

            last_log_likelihood = log_likelihood

            self.mu_all.append(self.mu)
            self.sigma_all.append(self.sigma)


        return self.mu, self.sigma, self.pi, last_log_likelihood

    def checkConvergence(self, last_log_likelihood, log_likelihood):
        #check the convergence of the algorithm
        return np.abs(log_likelihood - last_log_likelihood) < 1e-6

    def getLogGaussianProb(self, x, mean, cov):
        #calculate the log probability of a gaussian distribution
        #check if the covariance matrix is singular

        while np.linalg.det(cov) <= 0:
            cov += self.bias * np.identity(self.dim)

        det_cov = np.linalg.det(cov)

        return -0.5 * (np.dot(np.dot((x-mean).T, np.linalg.inv(cov)), (x-mean)) + np.log(det_cov) + self.dim * np.log(2*np.pi))


    def get_all_mu_sigma(self):
        return self.mu_all, self.sigma_all





