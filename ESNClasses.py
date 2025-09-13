import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import math
from scipy.sparse.linalg import eigs, ArpackNoConvergence
import os
import numpy as np
from scipy.special import erf



class Module(object):
    def __init__(self, *_args, seed=None, rnd=None, dtype=np.float64, **_kwargs):
        if rnd is None:
            self.rnd = np.random.default_rng(seed)
        else:
            self.rnd = rnd
        self.dtype = dtype

class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int,
                 bound: float = None, LowboundBias: float = None, HighboundBias: float = None, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bound is None:
            bound = math.sqrt(1 / input_dim)

        self.weight = self.rnd.uniform(low=-bound, high=bound, size=(output_dim, input_dim)).astype(self.dtype)
         # Generate bias values using positive and negative intervals
        if LowboundBias is not None and HighboundBias is not None:
            num_samples = output_dim // 2
            bias_interval1 = self.rnd.uniform(low=-HighboundBias, high=-LowboundBias, size=(num_samples,))
            bias_interval2 = self.rnd.uniform(low=LowboundBias, high=HighboundBias, size=(output_dim - num_samples,))
            bias_combined = np.concatenate((bias_interval1, bias_interval2))
            self.rnd.shuffle(bias_combined)  # Shuffle the values
            self.bias = bias_combined.astype(self.dtype)
        else:
            raise ValueError("LowboundBias and HighboundBias must be provided for bias generation.")

    def __call__(self, x: np.ndarray, beta: float = 0, winFactor : float=1):
        x = np.asarray(x)
        out = winFactor*np.matmul(x, self.weight.swapaxes(-1, -2)) + self.bias * (beta)
        return out


class ESN(Module):
    def __init__(self, dim: int, sr: float = 1.0, f=np.tanh,
                 a: float | None = None, p: float = 1.0, wpFactor: float = 1.0,  wnetFactor: float = 1.0, knet: float = 1.0, kbias : float = 1.0, theBiasFactor: float = 1.0, shrinkingFactor : float = 1.0, divideStates : float = 1.0,
                 init_state: np.ndarray | None = None, normalize: bool = True, **kwargs):
        super(ESN, self).__init__(**kwargs)
        self.dim = dim
        self.sr = sr
        self.f = f
        self.a = a
        self.p = p
        self.wpFactor = wpFactor
        self.wnetFactor = wnetFactor
        self.knet = knet
        self.kbias = kbias

        self.theBiasFactor = theBiasFactor
        self.theBias = self.rnd.uniform(low = -self.kbias, high = self.kbias, size=(self.dim,))

        self.shrinkingFactor = shrinkingFactor
        self.divideStates = divideStates

        self.x_init = np.zeros(dim, dtype=self.dtype)

        self.x = np.array(self.x_init)

        while True:
            try:
                #self.w_net = self.rnd.normal(loc=0, scale=1, size=(self.dim, self.dim)).astype(self.dtype)
                self.w_net = self.rnd.uniform(low=-self.knet, high=self.knet, size=(self.dim, self.dim)).astype(self.dtype)
                #self.w_net = (self.w_net  + self.w_net.T) / 2

                if self.p < 1.0:
                    w_con = np.full((dim * dim,), False)
                    w_con[:int(dim * dim * self.p)] = True
                    w_con = w_con.reshape((dim, dim))
                    self.rnd.shuffle(w_con)
                    self.w_net = self.w_net * w_con

                if normalize:
                    eigen_values = eigs(self.w_net, return_eigenvectors=False,
                                        k=min(self.dim, 2), which="LM", v0=np.ones(self.dim))

                    spectral_radius = max(abs(eigen_values))
                    #print("le spectral_radius =", spectral_radius)
                    self.w_net = (self.w_net / spectral_radius) * self.sr
                break
            except ArpackNoConvergence:
                continue


    def transform_reservoir_state(self, x):
       # x_transformed = np.copy(x)
       # x_transformed[::2] = (x[::2])   #even nodes are unchanged
       # x_transformed[1::2] = x[1::2]**2#odd nodes are squared

        x_transformed = np.copy(x)
        x_transformed[1::2] = np.abs(x_transformed[1::2]) *x_transformed[1::2]

        return x_transformed

    def __call__(self, x: np.ndarray, v: np.ndarray | None = None, r_param: float = None):

        x_next = self.wnetFactor*(np.matmul(self.x, (self.w_net).swapaxes(-1, -2)))
        if v is not None:
          x_next += v

        x_next += self.theBiasFactor * self.theBias
        #x_next *= self.shrinkingFactor

        x_next = self.transform_reservoir_state(x_next) #2
        x_next = self.f(x_next)


        if self.a is None:
            return x_next
        else:
            return (1 - self.a) * x + self.a * x_next

    def step(self, v: np.ndarray | None = None, p: float = None):
        self.x = self(self.x, v, p)
        #print("Max abs weight in reservoir:", np.max(np.abs(self.w_net)))


class RidgeReadout(Linear):
    def __init__(self, *args, lmbd: float = 0.0, **kwargs):
        super(RidgeReadout, self).__init__(*args, **kwargs)
        self.lmbd = lmbd

    def train(self, x: np.ndarray, y: np.ndarray):
        assert (x.ndim > 1) and (x.shape[-1] == self.input_dim)
        assert (y.ndim > 1) and (y.shape[-1] == self.output_dim)
        x_biased = np.ones((*x.shape[:-1], x.shape[-1] + 1), dtype=self.dtype)
        x_biased[..., 1:] = x
        xtx = x_biased.swapaxes(-2, -1) @ x_biased
        xty = x_biased.swapaxes(-2, -1) @ y
        sol = np.matmul(np.linalg.pinv(xtx + self.lmbd * np.eye(x.shape[-1] + 1)), xty)
        self.weight = sol[..., 1:, :].swapaxes(-2, -1)
        self.bias = sol[..., :1, :]
        return self.weight, self.bias

def createMyBifurcationDiagram(timeSeriesData, dictBifurcationData, bifurcationParameterValue):
  #timeSeriesData : the data associated to a value of the bifurcation parameter
  #dictBifurcationData : the dictionary storing all the bifurcation parameter values and the associated values to plot on the bifurcation diagram

  # Trouve les minima locaux (les indices des minima)
  ESNpredictions_minima_indices, _ = find_peaks(-timeSeriesData)  # Inverse pour détecter les minima

  # Récupérer les valeurs des minima pour chaque série
  ESNpredictions_minima_values = timeSeriesData[ESNpredictions_minima_indices]

  dictBifurcationData[bifurcationParameterValue] = ESNpredictions_minima_values

  return dictBifurcationData


def plotBifurcation(bifurcation_data, ax=None, myColor='blue'):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,6))

    xs, ys = [], []
    # collecte des abscisses et ordonnées
    for b, minima in bifurcation_data.items():
        m = np.asarray(minima).ravel()
        xs.extend([b] * m.size)
        ys.extend(m.tolist())

    # vérification des longueurs
    if len(xs) != len(ys):
        raise ValueError(f"Length mismatch: xs has {len(xs)} points, ys has {len(ys)}.")

    ax.scatter(xs, ys, s=5, color=myColor, label='Prediction', zorder=2)
    return ax

def plot_target_bifurcation(ax, data, cols, loads, nbMeasures, deltaLoad, num_points):
    xs, ys = [], []
    col = 21
    for load in loads:
        start = int(((load - 100) / deltaLoad) * nbMeasures)
        r = data[start:start+num_points, col] / 1e3
        idxs, _ = find_peaks(-r)
        xs.extend([load] * idxs.size)
        ys.extend(r[idxs].tolist())

    # Tracé sans label (pas dans la légende)
    ax.scatter(xs, ys, s=5, color='red', alpha=0.05, zorder=1)

    return ax