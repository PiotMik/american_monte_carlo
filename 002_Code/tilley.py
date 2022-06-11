from derivatives import AmericanOption, InterestRateSwap
import numpy as np
from itertools import groupby
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
sns.set()


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


def sort(sequence_to_sort_by, reversed, *args, **kwargs):
    mult = -1.0 if reversed else 1.0
    order = np.argsort(mult*sequence_to_sort_by)
    ordered_args = (arg[order, :] for arg in args)
    return ordered_args


def tilley_step(I, H, V, x, y, Q, P, rf_sim, epoch, derivative: Union[AmericanOption,
                                                                      InterestRateSwap], disc_fct):
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
    if type(derivative).__name__ == "AmericanOption":
        if derivative.option_type == "put":
            reverse = True
        else:
            reverse = False
    else:
        reverse = False

    rf_sim, V, I, H, x, y = sort(rf_sim[:, epoch], reverse,
                                 rf_sim, V, I, H, x, y)

    # 2 - compute I[k, t]
    I[:, epoch] = derivative.intrinsic_value(underlying_sim=rf_sim, t=epoch)

    # 3 - Create splits in Q bundles of P paths
    bundles = [(i * P, (i + 1) * P) for i in range(Q)]

    # 4 - Holding value
    if epoch == -1:
        for (l, r) in bundles:
            H[l:r, epoch] = I[l:r, epoch]
    else:
        for (l, r) in bundles:
            H[l:r, epoch] = disc_fct * V[l:r, epoch + 1].mean() + derivative.cashflow(underlying_sim=rf_sim[l:r, :],
                                                                                      t=epoch)

    if derivative.early_exercisable:
        # 5 - make a tentative exercise decision
        x[:, epoch] = I[:, epoch] >= H[:, epoch]

        # 6 - determine the sharp boundary
        sequence = x[:, epoch]
        k_star = boundary(sequence)

        # 7 - exercise indicator
        y[:, epoch] = np.array([0 if i != k_star else 1 for i, _ in enumerate(sequence)]).cumsum()
    else:
        x[:, epoch] = False
        y[:, epoch] = [1.0, 0.0][epoch == -rf_sim.shape[1] + 1]

    # 8 - current value of the derivative V[k, t]
    if not derivative.early_exercisable:
        V[:, epoch] = H[:, epoch]
    elif epoch == -1:
        V[:, epoch] = I[:, epoch]
    else:
        V[:, epoch] = y[:, epoch] * I[:, epoch] + (1 - y[:, epoch]) * H[:, epoch]

    return I, H, V, x, y, rf_sim


def tilley_price(rfr, time_steps, Q, P, derivative, rf_sim):

    paths = rf_sim.shape[0]
    T = derivative.expiry

    dt = T / (time_steps - 1)
    H = np.zeros_like(rf_sim)
    V = np.zeros_like(rf_sim)
    I = np.zeros_like(rf_sim)
    x = np.zeros_like(rf_sim)
    y = np.zeros_like(rf_sim)

    ## Tilley algorithm
    for epoch in range(-1, -time_steps - 1, -1):
        I, H, V, x, y, rf_sim = tilley_step(I, H, V, x, y, Q, P, rf_sim, epoch=epoch, derivative=derivative,
                                            disc_fct=np.exp(-rfr * dt))

    z = (y.cumsum(axis=1).cumsum(axis=1) == 1.0).astype(int)
    D = (np.exp(rfr*dt)*np.ones_like(I)).cumprod(axis=1)/np.exp(rfr*dt)
    price = (z[:, 1:]/D[:, 1:]*I[:, 1:]).sum()/paths if derivative.early_exercisable else (z/D*V).sum()/paths
    V[:, 0] = price

    early_exercise_prices = y*rf_sim*(I > H)
    early_exercise_prices[early_exercise_prices == 0.0] = -np.inf
    early_exercise_boundary = np.nanmax(early_exercise_prices, axis=0)
    early_exercise_boundary[np.isinf(early_exercise_boundary)] = np.nan
    return price, V, early_exercise_boundary


if __name__ == "__main__":
    pass
