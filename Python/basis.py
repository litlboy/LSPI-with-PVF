"""
This code was written by Sephora M in her project Graph ML ('https://github.com/Sephora-M/graph-rl/blob/master/lspi/basis_functions.py'). 
I simply modified the Proto Value Basis to adapt it to the continuous case.
"""

# -*- coding: utf-8 -*-
"""Abstract Base Class for Basis Function and some common implementations."""

import abc
import numpy as np
from scipy.spatial import cKDTree
from utils import build_similarity_graph, build_laplacian


class BasisFunction(object):

    r"""ABC for basis functions used by LSPI Policies.
    A basis function is a function that takes in a state vector and an action
    index and returns a vector of features. The resulting feature vector is
    referred to as :math:`\phi` in the LSPI paper (pg 9 of the PDF referenced
    in this package's documentation). The :math:`\phi` vector is dotted with
    the weight vector of the Policy to calculate the Q-value.
    The dimensions of the state vector are usually smaller than the dimensions
    of the :math:`\phi` vector. However, the dimensions of the :math:`\phi`
    vector are usually much smaller than the dimensions of an exact
    representation of the state which leads to significant savings when
    computing and storing a policy.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def evaluate(self, state, action):
        r"""Calculate the :math:`\phi` matrix for the given state-action pair.
        The way this value is calculated depends entirely on the concrete
        implementation of BasisFunction.
        Parameters
        ----------
        state : numpy.array
            The state to get the features for.
            When calculating Q(s, a) this is the s.
        action : int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.
        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.
        """
        pass  # pragma: no cover

    @abc.abstractproperty
    def num_actions(self):
        """Return number of possible actions.
        Returns
        -------
        int
            Number of possible actions.
        """
        pass  # pragma: no cover

    @staticmethod
    def _validate_num_actions(num_actions):
        """Return num_actions if valid. Otherwise raise ValueError.
        Return
        ------
        int
            Number of possible actions.
        Raises
        ------
        ValueError
            If num_actions < 1
        """
        if num_actions < 1:
            raise ValueError('num_actions must be >= 1')
        return num_actions

class RadialBasisFunction(BasisFunction):

    r"""Gaussian Multidimensional Radial Basis Function (RBF).
    Given a set of k means :math:`(\mu_1 , \ldots, \mu_k)` produce a feature
    vector :math:`(1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
    e^{-\gamma || s - \mu_k ||^2})` where `s` is the state vector and
    :math:`\gamma` is a free parameter. This vector will be padded with
    0's on both sides proportional to the number of possible actions
    specified.
    Parameters
    ----------
    means: list(numpy.array)
        List of numpy arrays representing :math:`(\mu_1, \ldots, \mu_k)`.
        Each :math:`\mu` is a numpy array with dimensions matching the state
        vector this basis function will be used with. If the dimensions of each
        vector are not equal than an exception will be raised. If no means are
        specified then a ValueError will be raised
    gamma: float
        Free parameter which controls the size/spread of the Gaussian "bumps".
        This parameter is best selected via tuning through cross validation.
        gamma must be > 0.
    num_actions: int
        Number of actions. Must be in range [1, :math:`\infty`] otherwise
        an exception will be raised.
    Raises
    ------
    ValueError
        If means list is empty
    ValueError
        If dimensions of each mean vector do not match.
    ValueError
        If gamma is <= 0.
    ValueError
        If num_actions is less than 1.
    Note
    ----
    The numpy arrays specifying the means are not copied.
    """

    def __init__(self, means, gamma, num_actions):
        """Initialize RBF instance."""
        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self.name = 'RBF'

        if len(means) == 0:
            raise ValueError('You must specify at least one mean')

        self.means = means

        if gamma <= 0:
            raise ValueError('gamma must be > 0')

        self.gamma = gamma

    @staticmethod
    def __check_mean_size(left, right):
        """Apply f if the value is not None.
        This method is meant to be used with reduce. It will return either the
        right most numpy array or None if any of the array's had
        differing sizes. I wanted to use a Maybe monad here,
        but Python doesn't support that out of the box.
        Return
        ------
        None or numpy.array
            None values will propogate through the reduce automatically.
        """
        if left is None or right is None:
            return None
        else:
            if left.shape != right.shape:
                return None
        return right

    def size(self):
        r"""Calculate size of the :math:`\phi` matrix.
        The size is equal to the number of means + 1 times the number of
        number actions.
        Returns
        -------
        int
            The size of the phi matrix that will be returned from evaluate.
        """
        return (len(self.means) + 1) * self.num_actions

    def evaluate(self, state):
        r"""Calculate the :math:`\phi` matrix.
        Matrix will have the following form:
        :math:`[\cdots, 1, e^{-\gamma || s - \mu_1 ||^2}, \cdots,
        e^{-\gamma || s - \mu_k ||^2}, \cdots]`
        where the matrix will be padded with 0's on either side depending
        on the specified action index and the number of possible actions.
        Returns
        -------
        numpy.array
            The :math:`\phi` vector. Used by Policy to compute Q-value.
        Raises
        ------
        IndexError
            If :math:`0 \le action < num\_actions` then IndexError is raised.
        ValueError
            If the state vector has any number of dimensions other than 1 a
            ValueError is raised.
        """

        if state.shape != self.means[0].shape:
            raise ValueError('Dimensions of state must match '
                             'dimensions of means')

        phi = np.zeros((self.num_actions, self.size()))
    

        rbf = [RadialBasisFunction.__calc_basis_component(state,
                                                          mean,
                                                          self.gamma)
               for mean in self.means]
        
        for action in range(self.num_actions):
            offset = (len(self.means)+1)*action
            phi[action, offset] = 1.
            phi[action, offset+1:offset+1+len(rbf)] = rbf

        return phi

    @staticmethod
    def __calc_basis_component(state, mean, gamma):
        mean_diff = state - mean
        return np.exp(-gamma*np.sum(mean_diff*mean_diff))

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.
        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.
        Raises
        ------
        ValueError
            If value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value
        
class ProtoValueBasis(BasisFunction):

    """Proto-value basis functions.
    These basis functions are formed using the eigenvectors of the graph Laplacian
    on an undirected graph formed from state transitions induced by the MDP
    Parameters
    ----------
    graph: pygsp.graphs
        Graph where the nodes are the states and the edges represent transitions
    num_actions: int
        Number of possible actions.
    num_laplacian_eigenvectors
    """

    def __init__(self, subsampled_states, var, nn, num_actions):
        """Initialize ExactBasis."""

        self.__num_actions = BasisFunction._validate_num_actions(num_actions)
        self.name = 'PVF'
        self.tab = subsampled_states
        self.var = var
        self.nn = nn
        self._compute_pvfs()

    def _compute_pvfs(self):
        self.ktree = cKDTree(self.tab, leafsize=50)
        kwargs = {'var':self.var, 'k':self.nn}
        self.W = build_similarity_graph(np.array(self.tab), **kwargs)
        laplacian = build_laplacian(self.W, laplacian_normalization="sym")
        diag_res = np.linalg.eig(laplacian)
        eigenvals = np.real(diag_res[0])
        indexes = np.argsort(eigenvals)
        self.eigenvecs = np.real(diag_res[1])[:, indexes]
        self.eigenvals = eigenvals[indexes]

    def set_num_features(self, n):
        self.k = n

    def size(self):
        r"""Return the vector size of the basis function.
        Returns
        -------
        int
            The size of the :math:`\phi` vector.
            (Referred to as k in the paper).
        """
        return (self.k) * self.__num_actions

    def evaluate(self, state):
        r"""Return a :math:`\phi` vector that has a self._num_laplacian_eigenvectors non-zero value.
        Parameters
        ----------
        state: numpy.array
            The state to get the features for. When calculating Q(s, a) this is
            the s.
        action: int
            The action index to get the features for.
            When calculating Q(s, a) this is the a.
        Returns
        -------
        numpy.array
            :math:`\phi` vector
        """
        distances, neighbors = self.ktree.query(state, k=self.nn)
        weights = np.exp(-distances / self.var)
        s = weights.sum()
        d = np.sum(self.W[neighbors, :], axis=1)
        v = weights/(d*s)**0.5
        lambdas = 1/(1-self.eigenvals[1:self.k+1])
        phis = np.array([self.eigenvecs[state, 1:self.k+1] for state in neighbors])*lambdas
        interp = np.dot(phis.T, v)
        return np.tensordot(np.eye(self.__num_actions), interp, axes=0).reshape((-1, self.size()))

    @property
    def num_actions(self):
        """Return number of possible actions."""
        return self.__num_actions

    @num_actions.setter
    def num_actions(self, value):
        """Set the number of possible actions.
        Parameters
        ----------
        value: int
            Number of possible actions. Must be >= 1.
        Raises
        ------
        ValueError
            if value < 1.
        """
        if value < 1:
            raise ValueError('num_actions must be at least 1.')
        self.__num_actions = value
