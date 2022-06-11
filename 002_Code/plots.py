from utils import plot_distribution, plot_early_exercise_boundary, gbm
from derivatives import AmericanOption, InterestRateSwap
from tilley import tilley_price
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(["science"])
plt.rcParams.update({'figure.figsize': (8, 4),
                     'axes.spines.top': False,
                     'axes.spines.right': False,
                     'xtick.top': False,
                     'ytick.right': False})
sns.set_context('notebook')


def plot_interest_rate_distribution():
    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 10000
    time_steps = 21

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    derivative = AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put")
    plot_distribution(rf_sim, T=derivative.expiry, epe=False,
                      quantiles=[0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
                      saveas=f"InterestRate.pdf")


def plot_tilley_exposures():
    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 10000
    time_steps = 21

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    derivatives = [AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put"),
                   InterestRateSwap(notional=1_000, fixed_rate=0.010501, expiry=T, position='receiver',
                                    payment_freq_pa=4.0)]

    alpha = [0.7]
    for derivative in derivatives:
        for a in alpha:
            P = int(paths**a)
            Q = int(paths/P)

            price_option, V_option, ex_bound = tilley_price(derivative=derivative,
                                                            time_steps=time_steps, P=P, Q=Q,
                                                            rf_sim=rf_sim[:P*Q, :], rfr=rfr)
            plot_distribution(V_option, quantiles=[0.025, 0.975], epe=True, T=derivative.expiry,
                              saveas=f"Tilley_Exposure_{type(derivative).__name__}.pdf")


def plot_tilley_early_exercise():
    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 10000
    time_steps = 21

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    derivative = AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put")
    alpha = 0.7
    P = int(paths ** alpha)
    Q = int(paths / P)

    price_option, V_option, ex_bound = tilley_price(derivative=derivative,
                                                    time_steps=time_steps, P=P, Q=Q,
                                                    rf_sim=rf_sim[:P * Q, :], rfr=rfr)
    plot_early_exercise_boundary(rf_sim, ex_bound=ex_bound, T = derivative.expiry,
                                 quantiles=[0.25, 0.75],
                                 saveas=f"Tilley_EarlyExercise_{type(derivative).__name__}.pdf")


def plot_tilley_alpha_dependency():
    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 50000
    time_steps = 21

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    derivatives = [AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put"),
                   InterestRateSwap(notional=1_000, fixed_rate=0.010501, expiry=T, position='receiver',
                                    payment_freq_pa=4.0)]

    for i, derivative in enumerate(derivatives):
        epes = {}
        if i == 0:
            alpha = np.linspace(0.5, 0.7, 15)[3:-3]
        else:
            alpha = np.linspace(0, 1, 8)[1:-2]
        for a in alpha:
            P = int(paths**a)
            Q = int(paths/P)

            price_option, V_option, ex_bound = tilley_price(derivative=derivative,
                                                            time_steps=time_steps, P=P, Q=Q,
                                                            rf_sim=rf_sim[:P*Q, :], rfr=rfr)
            epes[f"$\\alpha = {a:.4f}$"] = np.maximum(V_option, 0).mean(axis=0).T
        df_epes = pd.DataFrame.from_dict(epes)

        fig, ax = plt.subplots(figsize=(8, 4))
        df_epes.plot(linestyle="--", ax=ax)
        plt.show()
        fig.savefig(f"003_Outputs/Tilley_alpha_{type(derivative).__name__}.pdf")
        plt.close(fig)


if __name__ == "__main__":

    plot_interest_rate_distribution()
    plot_tilley_early_exercise()
    plot_tilley_exposures()
    plot_tilley_alpha_dependency()

    # if type(derivative).__name__ == "InterestRateSwap":
    #     cf_ind = (rf_sim.shape[1] - 1) / derivative.expiry / derivative.payment_freq_pa
    #     cf_ind = [x for x in range(1, time_steps) if x % cf_ind == 0]
    #     d_fct = np.array([np.exp(-rfr * x / (time_steps - 1) * derivative.expiry) for x in cf_ind])
    #     control_price = np.concatenate([
    #         (derivative.notional * (rf_sim[:, epoch] - derivative.fixed_rate) / derivative.payment_freq_pa).reshape(-1,
    #                                                                                                                 1)
    #         for epoch in cf_ind], axis=1)
    #     control_price = (control_price * d_fct).sum(axis=1).mean()
    # else:
    #     d1 = (np.log(r0 / derivative.strike) + (rfr - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    #     d2 = d1 - sigma * np.sqrt(T)
    #     if derivative.option_type == "call":
    #         control_price = norm.cdf(d1) * r0 - norm.cdf(d2) * derivative.strike * np.exp(-rfr * T)
    #     else:
    #         control_price = norm.cdf(-d2) * derivative.strike * np.exp(-rfr * T) - norm.cdf(-d1) * r0
    # print(f'Tilley price: {price_option:.8f}\n'
    #       f'Control price: {control_price:.8f}')
