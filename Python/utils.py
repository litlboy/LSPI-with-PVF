import numpy as np
import scipy.spatial.distance as sd
import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def collect_episodes(mdp, weight=None, basis=None, horizon=None, n_episodes=1, custom=None):
    dataset = []
    for _ in range(n_episodes):
        state = mdp.reset()
        for _ in range(horizon):
            if weight is None:
                action = mdp.action_space.sample()
            else:
                q = np.dot(basis.evaluate(state), weight)
                action = np.argmax(q)
            next_state, reward, terminal, _ = mdp.step(action)
            if custom and terminal:
                reward = custom
            dataset.append((state, action, reward, next_state))
            state = copy.copy(next_state)
            if terminal:
                # Finish rollout if terminal state reached
                break
                # We need to compute the empirical return for each time step along the
                # trajectory
    return dataset

def subsample(states, epsilon):
        l = []
        indexes = np.arange(len(states))
        i_0 = np.random.choice(indexes)
        l.append(i_0)
        x_0 = states[i_0]
        dists = sd.cdist([x_0], states)[0]
        mask = (dists > epsilon)
        while mask.sum() > 0:
            i_0 = np.random.choice(indexes[mask])
            l.append(i_0)
            mask[i_0] = 0
            x_0 = states[i_0]
            dists = sd.cdist([x_0], states)[0]
            mask = mask*(dists > epsilon)
        return [states[i] for i in l]

def build_rbf_centers(env, num_rbf):
    dim = env.env.observation_space.high.size
    num_ind = np.prod(num_rbf)
    # Parameters
    discrt = 4
    num_rbf = discrt * np.ones(dim).astype(int)
    width = 1. / (num_rbf - 1.)
    c = np.zeros((num_ind, dim))
    for i in range(num_ind):
        if i == 0:
            pad_num = dim
        else:
            pad_num = dim - int(np.log(i) / np.log(discrt)) - 1
        ind = np.base_repr(i, base=discrt, padding=pad_num)
        ind = np.asarray([float(j) for j in list(ind)])
        c[i, :] = width * ind
    return c

def build_similarity_graph(X, var=1, eps=0, k=0):
    assert eps + k != 0, "Choose either epsilon graph or k-nn graph"
    dists = sd.squareform(sd.pdist(X, "sqeuclidean"))
    similarities = np.exp(-dists / var)
    if eps:
        similarities[similarities < eps] = 0
    if k:
        sort = np.argsort(similarities)[:, ::-1]  # descending
        mask = sort[:, k + 1:]  # indices to mask
        for i, row in enumerate(mask): similarities[i, row] = 0
    np.fill_diagonal(similarities, 0)  # remove self similarity
    return (similarities + similarities.T) / 2  # make the graph undirected 

def build_laplacian(W, laplacian_normalization=""):
    degree = W.sum(1)
    if not laplacian_normalization:
        return np.diag(degree) - W
    elif laplacian_normalization == "sym":
        aux = np.diag(1 / np.sqrt(degree))
        return np.eye(*W.shape) - aux.dot(W.dot(aux))
    elif laplacian_normalization == "rw":
        return np.eye(*W.shape) - np.diag(1 / degree).dot(W)
    else: raise ValueError
        
def solve(A, b):
    info = {}
    w = None

    info['acond'] = np.linalg.cond(A)
    w = np.dot(np.linalg.pinv(A),b)

    return w,info

        
class ManifoldPlot(object):
    
    def __init__(self, pvfs, env, state_indexes, **args):
        self.state_indexes = state_indexes
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.env = env
        self.pvfs = pvfs
        self.args = args
        self._compute_phis()
        
    def _compute_phis(self):
        X = np.linspace(self.low[self.state_indexes[0]], self.high[self.state_indexes[0]], 10)
        Y = np.linspace(self.low[self.state_indexes[1]], self.high[self.state_indexes[1]], 10)
        X, Y = np.meshgrid(X, Y)
        self.phis = np.array([self.func(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
        self.X, self.Y = X, Y
        
    def func(self, x, y):
        s = np.zeros(self.env.observation_space.shape[0])
        s[self.state_indexes[0]] = x
        s[self.state_indexes[1]] = y
        return self.pvfs.evaluate(s)[:self.pvfs.k]
        
    def plot_embedding(self, eigens):
        fig = plt.figure(figsize=(10,10))
        n = len(eigens)
        rows = n//3 + 1
        for i, j in enumerate(eigens):
            ax = fig.add_subplot(rows, 3, i+1, projection='3d')
            zs = self.phis[:,j]
            Z = zs.reshape(self.X.shape)

            ax.plot_surface(self.X, self.Y, Z, cmap=cm.coolwarm)

            ax.set_xlabel(self.args['X'])
            ax.set_ylabel(self.args['Y'])
            ax.set_zlabel(self.args['Z'])
            ax.set_title('Laplacian basis function %s'%(j+1))
        plt.tight_layout()
