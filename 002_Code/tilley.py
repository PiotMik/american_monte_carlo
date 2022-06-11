from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from itertools import groupby
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union
sns.set()


def gbm(r0=0.05, sigma=0.2, mu=0.01, T=1.0, steps=5, paths=2):
    dt = T / steps
    Z = np.random.normal(size=(steps, paths))
    X = (mu - sigma ** 2 / 2) * dt + Z * sigma * np.sqrt(dt)
    X[0, :] = 0.0
    return r0 * np.exp(X.cumsum(axis=0))


@dataclass
class AmericanOption:
    early_exercisable = True
    strike: float
    expiry: float
    option_type: str

    def intrinsic_value(self, underlying_sim: np.array, t: int):
        ST = underlying_sim[:, t]
        mult = 1.0 if self.option_type.lower() == "call" else -1.0
        return np.maximum(mult * (ST - self.strike), 0)

    def cashflow(self, underlying_sim: np.array, t: int):
        return underlying_sim[:, t]*0.0


@dataclass
class InterestRateSwap:
    early_exercisable = False
    fixed_rate: float
    expiry: float
    payment_freq_pa: float
    position: str
    notional: float

    def intrinsic_value(self, underlying_sim: np.array, t: int):
        if t == -1:
            iv = self.cashflow(underlying_sim, t)
        else:
            iv = underlying_sim[:, t]*np.nan
        return iv

    def cashflow(self, underlying_sim: np.array, t: int):

        n_max = underlying_sim.shape[1]
        if -t == n_max:
            bpayment = 0.0
        elif (- t - 1) % int((underlying_sim.shape[1] - 1) / self.expiry / self.payment_freq_pa) == 0:
            bpayment = 1.0
        else:
            bpayment = 0.0

        rt = underlying_sim[:, t]
        mult = 1.0 if self.position.lower() == "receiver" else -1.0
        cf = self.notional * mult * (rt - self.fixed_rate) / self.payment_freq_pa
        return cf*bpayment


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
    rf_sim, V, I, H, x, y = sort(rf_sim[:, epoch], rf_sim, V, I, H, x, y)

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
    price = (z/D*I).sum()/paths if derivative.early_exercisable else (z/D*V).sum()/paths

    early_exercise_prices = z*rf_sim
    early_exercise_prices[early_exercise_prices == 0.0] = np.inf
    early_exercise_boundary = np.nanmin(early_exercise_prices, axis=0)
    early_exercise_boundary[np.isinf(early_exercise_boundary)] = np.nan
    return price, V, early_exercise_boundary


if __name__ == "__main__":

    rfr = 0.05
    sigma = 0.1
    r0 = 0.05
    K = 0.06
    T = 5.0

    P = 500
    Q = 100
    time_steps = 81

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=Q*P).T

    # derivative = AmericanOption(strike=K, expiry=T, option_type="call")
    derivative = InterestRateSwap(notional=1_000_000, fixed_rate=0.05, expiry=T, position='receiver',
                                  payment_freq_pa=2.0)
    price_option, V_option, ex_bound = tilley_price(derivative=derivative,
                                                    time_steps=time_steps, P=P, Q=Q, rf_sim=rf_sim, rfr=rfr)

    quantiles_to_plot = [0.75, 0.25]
    quantiles_option = pd.DataFrame(np.quantile(V_option, quantiles_to_plot, axis=0).T,
                                    columns=[str(q) for q in quantiles_to_plot],
                                    index=np.linspace(0, T, time_steps))
    quantiles_option['EPE'] = np.maximum(V_option, 0).mean(axis=0).T
    fig, ax = plt.subplots(2, 1)
    quantiles_option.drop('EPE', axis=1).plot(ax=ax[0])
    quantiles_option['EPE'].plot(ax=ax[0], linestyle="--", legend=True)
    ax[0].set_title(str(derivative))

    eeb = pd.DataFrame(ex_bound, index=quantiles_option.index,
                       columns=['Early Exercise Boundary']).replace(0.0, np.nan)
    eeb = eeb[np.abs(eeb.diff()) < 10].dropna()
    ax[1].plot(quantiles_option.index, rf_sim[np.random.choice(range(rf_sim.shape[0]), 300), :].T,
               color='grey', alpha=0.2)
    eeb.plot(ax=ax[1])

    plt.show()

    if type(derivative).__name__ == "InterestRateSwap":
        cf_ind = (rf_sim.shape[1] - 1)/derivative.expiry/derivative.payment_freq_pa
        cf_ind = [x for x in range(1, time_steps) if x % cf_ind == 0]
        d_fct = np.array([np.exp(-rfr*x/(time_steps-1)*derivative.expiry) for x in cf_ind])
        control_price = np.concatenate([
            (derivative.notional*(rf_sim[:, epoch] - derivative.fixed_rate)/derivative.payment_freq_pa).reshape(-1, 1)
            for epoch in cf_ind], axis=1)
        control_price = (control_price*d_fct).sum(axis=1).mean()
    else:
        d1 = (np.log(r0/K) + (rfr - sigma**2/2)*T)/(sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        control_price = norm.cdf(d1)*r0 - norm.cdf(d2)*K*np.exp(-rfr*T)

    print(f'Tilley price: {price_option:.8f}\n'
          f'Control price: {control_price:.8f}')
