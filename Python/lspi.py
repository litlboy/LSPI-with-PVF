import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import copy
from utils import solve

class LSPI(object):

    def __init__(self, data, basis, gamma, n_actions):

        self.size = len(data)
        self.basis = basis
        self.gamma = gamma
        self.n = n_actions

        self.states = [data[i][0] for i in range(self.size)]
        self.next_states = [data[i][3] for i in range(self.size)]
        self.actions = [data[i][1] for i in range(self.size)]
        self.rewards = np.array([data[i][2] for i in range(self.size)])

        self._compute_phis()

    def _compute_phis(self):

        n_cores = multiprocessing.cpu_count()

        # Build chunks
        chunks_states = {}
        chunks_nstates = {}
        for i in range(n_cores):
            chunks_states[i] = self.states[i*(self.size//n_cores):(i+1)*(self.size//n_cores)]
            chunks_nstates[i] = self.next_states[i*(self.size//n_cores):(i+1)*(self.size//n_cores)]
            if i == n_cores - 1:
                chunks_states[i] = self.states[i*(self.size//n_cores):]
                chunks_nstates[i] = self.next_states[i*(self.size//n_cores):]
        
        res_states = Parallel(n_jobs=n_cores)(delayed(self._basis_eval)(*chunks_states[i]) for i in range(n_cores))
        res_nstates = Parallel(n_jobs=n_cores)(delayed(self._basis_eval)(*chunks_nstates[i]) for i in range(n_cores))

        l_states = []
        l_nstates = []

        for chunk_s, chunks_ns in zip(res_states, res_nstates):
            l_states += chunk_s
            l_nstates += chunks_ns

        self.phi_state = np.array(l_states)
        self.phi_nstate = np.array(l_nstates)

        del self.states
        del self.next_states

    def _basis_eval(self, *args):
        return [self.basis.evaluate(s)for s in args]
    
    def iteration(self, theta):
        features = self.phi_state[np.arange(len(self.actions)), self.actions, :]
        next_actions = np.argmax(np.dot(self.phi_nstate, theta), axis=1)
        next_features = self.phi_nstate[np.arange(len(next_actions)), next_actions, :]
        self.A = np.sum(np.einsum('ij,ik->ijk',features, features - self.gamma*next_features), axis=0) \
                    + 0.001*np.eye(self.basis.size())
        self.b = np.sum(features*self.rewards[:, None], axis=0)
        return solve(self.A,self.b)[0]