from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from itertools import groupby
import pandas as pd
import multiprocessing
import matplotlib.pyplot as plt


def gbm(r0=0.05, sigma=0.2, mu=0.01, T=1.0, steps=5, paths=2):
    dt = T / steps
    Z = np.random.normal(size=(steps, paths))
    X = (mu - sigma ** 2 / 2) * dt + Z * sigma * np.sqrt(dt)
    X[0, :] = 0.0
    return r0 * np.exp(X.cumsum(axis=0))


@dataclass
class AmericanOption:
    strike: float
    expiry: float
    option_type: str

    def intrinsic_value(self, underlying_price: np.array):
        mult = 1.0 if self.option_type.lower() == "call" else -1.0
        return np.maximum(mult * (underlying_price - self.strike), 0)


def boundary(sequence: np.array):
    sequence_str = "".join(str(int(c)) for c in sequence)
    subsequences = [(x, len(''.join(g))) for x, g in groupby(sequence_str)]
    subsequences_df = pd.DataFrame(subsequences, columns=['Character', 'Seq_len'])
    subsequences_df.index = subsequences_df['Seq_len'].cumsum() - subsequences_df['Seq_len']
    subsequences_df.index.name = 'Position'
    subsequences_df['Auxiliary_column'] = subsequences_df['Seq_len'] * (1 - subsequences_df['Character'].astype(int))
    subsequences_df['LongestSubsequentZeros'] = subsequences_df['Auxiliary_column'][::-1].cummax()[::-1]
    subsequences_df.drop(['Auxiliary_column'], axis=1, inplace=True)
    subsequences_df['BeyondBoundary'] = subsequences_df['Seq_len'] > subsequences_df['LongestSubsequentZeros']
    ones_beyond_boundary = subsequences_df[np.logical_and(subsequences_df['Character'] == "1",
                                                          subsequences_df['BeyondBoundary'])]
    try:
        k_star = ones_beyond_boundary.head().index[0]
    except IndexError:
        k_star = len(sequence)
    return k_star


def sort(sequence_to_sort_by, *args):
    order = np.argsort(sequence_to_sort_by)
    ordered_args = (arg[order, :] for arg in args)
    return ordered_args


def tilley_step(I, H, V, x, y, Q, P, rf_sim, epoch, derivative, disc_fct):
    """
    Perform one step of the Tilley algorithm.

    Parameters
    ----------
    I: np.array
        Intrinsic value of the derivative.
    H: np.array
        Holding value of the derivative.
    V: np.array
        Price value of the derivative
    x: np.array
        Tentative decision if exercise american derivative.
    y: np.array
        Sharp decision if exercise american derivative.
    rf_sim: np.array
        (n_time_steps, n_paths) array, all above arrays need to be of the same shape.
    Q: int
        Number of bundles to group paths into.
    P: int
        Number of paths per bundle
    epoch: int
        Iterator, time step identifier. -1 -> last, -2 -> second to last, etc.
    derivative: object
        Object with intrinsic_value() method.
    disc_fct: float
        1-step discount factor

    Returns
    -------
    I, H, V, x, y, rf_sim
        Input arrays with updated values as per after "epoch" step.
    """

    # 1 - reorder by stock price
    rf_sim, V, I, H, x, y = sort(rf_sim[:, epoch], rf_sim, V, I, H, x, y)

    # 2 - compute I[k, t]
    I[:, epoch] = derivative.intrinsic_value(underlying_price=rf_sim[:, epoch])

    # 3 - Create splits in Q bundles of P paths
    bundles = [(i * P, (i + 1) * P) for i in range(Q)]

    # 4 - Holding value
    if epoch == -1:
        for (l, r) in bundles:
            H[l:r, epoch] = I[l:r, epoch]
    else:
        for (l, r) in bundles:
            H[l:r, epoch] = disc_fct * V[l:r, epoch + 1].mean()

    # 5 - make a tentative exercise decision
    x[:, epoch] = I[:, epoch] >= H[:, epoch]

    # 6 - determine the sharp boundary
    sequence = x[:, epoch]
    k_star = boundary(sequence)

    # 7 - exercise indicator
    y[:, epoch] = np.array([0 if i != k_star else 1 for i, _ in enumerate(sequence)]).cumsum()

    # 8 - current value of the option V[k, t]
    if epoch == -1:
        V[:, epoch] = I[:, epoch]
    else:
        V[:, epoch] = y[:, epoch] * I[:, epoch] + (1 - y[:, epoch]) * H[:, epoch]

    return I, H, V, x, y, rf_sim


def tilley_price(time_steps, Q, P, derivative, rf_sim):

    paths = rf_sim.shape[0]
    rfr = 0.01
    T = derivative.expiry

    dt = T / (time_steps - 1)
    H = np.zeros_like(rf_sim)
    V = np.zeros_like(rf_sim)
    I = np.zeros_like(rf_sim)
    x = np.zeros_like(rf_sim)
    y = np.zeros_like(rf_sim)


    ## Tilley algorithm
    for epoch in range(-1, -time_steps - 1, -1):
        I, H, V, x, y, rf_sim = tilley_step(I, H, V, x, y, Q, P, rf_sim,
                                            epoch=epoch, derivative=derivative, disc_fct=np.exp(-rfr * dt))

    z = (y.cumsum(axis=1).cumsum(axis=1) == 1.0).astype(int)
    D = (np.exp(rfr*dt)*np.ones_like(I)).cumprod(axis=1)/np.exp(rfr*dt)
    price = (z/D*I).sum()/paths
    return price, V


if __name__ == "__main__":

    rfr = 0.07
    sigma = 0.3
    r0 = 50.0
    K = 50.0
    T = 1.0

    P = 25
    Q = 25
    time_steps = 8
    np.random.seed(1)
    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=Q*P).T
    tilley_price, V = tilley_price(derivative=AmericanOption(strike=K, expiry=T, option_type="call"),
                                   time_steps=time_steps, P=P, Q=Q, rf_sim=rf_sim)

    quantiles = pd.DataFrame(np.quantile(V, [0.95, 0.75, 0.5, 0.25, 0.05], axis=0).T,
                             columns=['0.95', '0.75', '0.5', '0.25', '0.05'],
                             index=np.linspace(0, T, time_steps))
    fig, ax = plt.subplots()
    quantiles.plot(ax=ax)
    plt.show()

    d1 = (np.log(r0/K) + (rfr - sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_price = norm.cdf(d1)*r0 - norm.cdf(d2)*K*np.exp(-rfr*T)

    print(f'Tilley price: {tilley_price:.8f}\n'
          f'BS price: {bs_price:.4f}')

