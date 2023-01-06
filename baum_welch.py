import numpy as np
from scipy.stats import multivariate_normal

# A = transition matrix, X = observation matrix, pi = initial state distribution
# mu = mean of distribution, sigma2 = sigma^2 of distribution
def e_step(X, pi, A, mu, sigma2):
    """E-step: forward-backward message passing"""
    # Messages and sufficient statistics
    # N = number of training sets, T = number of time steps
    # K = length of feature vector, M = number of hidden states
    N, T, K = X.shape
    M = A.shape[0]
    alpha = np.zeros([N,T,M])  # [N,T,M]
    alpha_sum = np.zeros([N,T])  # [N,T], normalizer for alpha
    beta = np.zeros([N,T,M])  # [N,T,M]
    gamma = np.zeros([N,T,M])  # [N,T,M]
    xi = np.zeros([N,T-1,M,M])  # [N,T-1,M,M]

    # Forward messages
    for m in range(M):
        alpha[:,0,m] = multivariate_normal.pdf([X[:,0,:]], mu[m,:], sigma2[m]) * pi[m]
    alpha_sum[:,0] = np.sum(alpha[:,0,:],axis=1)
    alpha[:,0,:] = alpha[:,0,:] / alpha_sum[:,0].reshape(len(alpha_sum[:,0]),1)

    for t in range(1,T):
        for m in range(M): # zt state
            alpha[:,t,m] = np.dot(A[:,m], alpha[:,t-1,:].T) * multivariate_normal.pdf([X[:,t,:]],
                                                                                      mu[m,:],
                                                                                      sigma2[m])
        alpha_sum[:,t] = np.sum(alpha[:,t,:],axis=1)
        alpha[:,t,:] = alpha[:,t,:] / alpha_sum[:,t].reshape(len(alpha_sum[:,t]),1)

    # Backward messages
    beta[:,-1,:] = 1

    # t = T ... 2
    for t in range(T-1,0,-1):
        for m1 in range(M): # zt - 1
            temp_sum = 0
            for m2 in range(M): # zt
                temp_sum = temp_sum + (A[m1,m2] * multivariate_normal.pdf([X[:,t,:]], mu[m2,:], sigma2[m2]) * beta[:,t,m2])
            beta[:,t-1,m1] = temp_sum / alpha_sum[:,t]

    # Sufficient statistics
    gamma = alpha * beta

    # pairwise
    for t in range(1,T):
        for m1 in range(M): # zt - 1
            for m2 in range(M): # zt
                xi[:,t-1,m1,m2] = (A[m1,m2] * multivariate_normal.pdf([X[:,t,:]], mu[m2,:], sigma2[m2]) * alpha[:,t-1,m1] * beta[:,t,m2]) / alpha_sum[:,t]

    # Although some of them will not be used in the M-step, please still
    # return everything as they will be used in test cases
    return alpha, alpha_sum, beta, gamma, xi


def m_step(X, gamma, xi):
    """M-step: MLE"""
    # TODO ...
    # pi update
    pi = np.mean(gamma[:,0,:],axis=0)

    # A update
    A = np.mean(np.sum(xi,axis=1),axis=0)
    A = A / A.sum(axis=-1, keepdims=True)

    # mu update
    mu = np.zeros((gamma.shape[2],X.shape[2]))
    for m in range(A.shape[0]):
        num = np.mean(np.sum(np.multiply(gamma[:,:,m].reshape(gamma[:,:,m].shape[0],gamma[:,:,m].shape[1],1),X),axis=1),axis=0)
        denom = np.mean(np.sum(gamma[:,:,m],axis=1))
        mu[m,:] = num / denom

    # sigma update
    sigma2 = np.zeros(gamma.shape[2])
    for m in range(A.shape[0]):
        n_temp = []
        for n in range(X.shape[0]):
            t_temp = []
            for t in range(X.shape[1]):
                t_temp.append((np.linalg.norm((X[n,t,:] - mu[m,:]))**2) * gamma[n,t,m])
            n_temp.append(np.sum(t_temp))
        n_temp = np.asarray(n_temp)
        denom = np.sum(gamma[:,:,m],axis=1) * X.shape[2]
        sigma2[m] = np.mean(n_temp) / np.mean(denom)

    return pi, A, mu, sigma2


def hmm_train(X, pi, A, mu, sigma2, em_step=20):
    """Run Baum-Welch algorithm."""
    for step in range(em_step):
        _, alpha_sum, _, gamma, xi = e_step(X, pi, A, mu, sigma2)
        pi, A, mu, sigma2 = m_step(X, gamma, xi)
        print(f"step: {step}  ln p(x): {np.einsum('nt->', np.log(alpha_sum))}")
    return pi, A, mu, sigma2


def hmm_generate_samples(N, T, pi, A, mu, sigma2):
    """Given pi, A, mu, sigma2, generate [N,T,K] samples."""
    M, K = mu.shape
    Y = np.zeros([N,T], dtype=int)
    X = np.zeros([N,T,K], dtype=float)
    for n in range(N):
        Y[n,0] = np.random.choice(M, p=pi)  # [1,]
        X[n,0,:] = multivariate_normal.rvs(mu[Y[n,0],:], sigma2[Y[n,0]] * np.eye(K))  # [K,]
    for t in range(T - 1):
        for n in range(N):
            Y[n,t+1] = np.random.choice(M, p=A[Y[n,t],:])  # [1,]
            X[n,t+1,:] = multivariate_normal.rvs(mu[Y[n,t+1],:], sigma2[Y[n,t+1]] * np.eye(K))  # [K,]
    return X


def main():
    """Run Baum-Welch on a simulated toy problem."""
    # Generate a toy problem
    np.random.seed(12345)  # for reproducibility
    N, T, M, K = 10, 100, 4, 2
    pi = np.array([.0, .0, .0, 1.])  # [M,]
    A = np.array([[.7, .1, .1, .1],
                  [.1, .7, .1, .1],
                  [.1, .1, .7, .1],
                  [.1, .1, .1, .7]])  # [M,M]
    mu = np.array([[2., 2.],
                   [-2., 2.],
                   [-2., -2.],
                   [2., -2.]])  # [M,K]
    sigma2 = np.array([.2, .4, .6, .8])  # [M,]
    X = hmm_generate_samples(N, T, pi, A, mu, sigma2)

    # Run on the toy problem
    pi_init = np.random.rand(M)
    pi_init = pi_init / pi_init.sum()
    A_init = np.random.rand(M, M)
    A_init = A_init / A_init.sum(axis=-1, keepdims=True)
    mu_init = 2 * np.random.rand(M, K) - 1
    sigma2_init = np.ones(M)
    pi, A, mu, sigma2 = hmm_train(X, pi_init, A_init, mu_init, sigma2_init, em_step=20)
    print(pi)
    print(A)
    print(mu)
    print(sigma2)


if __name__ == '__main__':
    main()
