from utils import gbm
from derivatives import AmericanOption, InterestRateSwap
from tilley import tilley_price
from tsitsiklis import tsitsiklis_price
import numpy as np
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

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

    paths = 1000
    time_steps = 51

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    qs = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975]
    df_quantiles = pd.DataFrame(np.quantile(rf_sim, qs, axis=0).T,
                                columns=['$q_{' + f'{q:.3f}' + "}$" for q in qs],
                                index=np.linspace(0, T, rf_sim.shape[1]))
    fig, ax = plt.subplots()
    ax.plot(df_quantiles.index, rf_sim.T, color='grey', alpha=0.4)
    df_quantiles.plot(ax=ax, colormap='jet')
    fig.savefig(f"003_Outputs/InterestRate.pdf")
    plt.show()
    plt.close(fig)


def plot_exposures():
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

    for derivative, alpha in zip(derivatives, [0.5, 0.65]):
        P = int(paths**alpha)
        Q = int(paths/P)

        price_tilley, V_tilley, _ = tilley_price(derivative=derivative,
                                                 time_steps=time_steps, P=P, Q=Q,
                                                 rf_sim=rf_sim[:P*Q, :], rfr=rfr)
        price_tsitsiklis, V_tsitsiklis, _ = tsitsiklis_price(derivative=derivative,
                                                             rf_sim=rf_sim,
                                                             rfr=rfr, time_steps=time_steps)

        fig, ax = plt.subplots()
        quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]
        for V_estimate, name, line in zip([V_tilley, V_tsitsiklis], ['Tilley', 'Tsitsiklis'], ['-', '--']):
            df_quantiles = pd.DataFrame(np.quantile(V_estimate, quantiles, axis=0).T,
                                        columns=[name + ' $q_{' + f'{q:.3f}' + "}$" for q in quantiles],
                                        index=np.linspace(0, T, V_estimate.shape[1]))

            df_quantiles.plot(ax=ax, linestyle=line, colormap='jet')
            #df_quantiles['EPE'] = np.maximum(V_estimate, 0).mean(axis=0).T
            #df_quantiles['EPE'].plot(ax=ax, legend=True)

        plt.show()
        fig.savefig(f"003_Outputs/Exposures_{type(derivative).__name__}.pdf")
        plt.close(fig)


def plot_epe():
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

    for derivative, alpha in zip(derivatives, [0.5, 0.65]):
        P = int(paths**alpha)
        Q = int(paths/P)

        price_tilley, V_tilley, _ = tilley_price(derivative=derivative,
                                                 time_steps=time_steps, P=P, Q=Q,
                                                 rf_sim=rf_sim[:P*Q, :], rfr=rfr)
        price_tsitsiklis, V_tsitsiklis, _ = tsitsiklis_price(derivative=derivative,
                                                             rf_sim=rf_sim,
                                                             rfr=rfr, time_steps=time_steps)
        fig, ax = plt.subplots()
        quantiles = [0.025, 0.1, 0.5, 0.9, 0.975]
        for V_estimate, name, line in zip([V_tilley, V_tsitsiklis], ['Tilley', 'Tsitsiklis'], ['-', '--']):
            df_quantiles = pd.DataFrame(np.quantile(V_estimate, quantiles, axis=0).T,
                                        columns=[name + ' $q_{' + f'{q:.3f}' + "}$" for q in quantiles],
                                        index=np.linspace(0, T, V_estimate.shape[1]))

            df_quantiles[f'{name} EPE'] = np.maximum(V_estimate, 0).mean(axis=0).T
            df_quantiles[f'{name} EPE'].plot(ax=ax, legend=True, linestyle=line)

        plt.show()
        fig.savefig(f"003_Outputs/EPEs_{type(derivative).__name__}.pdf")
        plt.close(fig)


def plot_early_exercise():
    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 5000
    time_steps = 51

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    derivative = AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put")
    alpha = 0.5
    P = int(paths ** alpha)
    Q = int(paths / P)

    _, V_tilley, eeb_tilley = tilley_price(derivative=derivative,
                                           time_steps=time_steps, P=P, Q=Q,
                                           rf_sim=rf_sim[:P * Q, :], rfr=rfr)
    _, V_tsitsiklis, eeb_tsitsiklis = tsitsiklis_price(derivative=derivative,
                                                       rf_sim=rf_sim,
                                                       rfr=rfr, time_steps=time_steps)

    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, T, V_tsitsiklis.shape[1]),
            rf_sim[np.random.choice(range(rf_sim.shape[0]), 1000), :].T,
            color='grey', alpha=0.2)
    ax.set_ylabel('$r_t$')
    ax.set_xlabel('$t$')

    for V_estimate, eeb_estimate, name in zip([V_tilley, V_tsitsiklis],
                                              [eeb_tilley, eeb_tsitsiklis],
                                              ['Tilley', 'Tsitsiklis']):
        eeb_estimate[eeb_estimate > rf_sim.mean(axis=0)] = np.nan
        eeb = pd.DataFrame(eeb_estimate, index=np.linspace(0, T, V_estimate.shape[1]),
                           columns=[f'{name} Early Exercise Boundary']).replace(0.0, np.nan)
        eeb.plot(ax=ax)
    plt.show()
    fig.savefig(f"003_Outputs/EarlyExerciseBoundaries_AmericanOption.pdf")
    plt.close(fig)


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


def plot_tsitsiklis_regressions():
    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 10000
    time_steps = 21

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T
    derivative = AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put")

    V = np.zeros_like(rf_sim)
    cf = np.zeros_like(rf_sim)
    T = derivative.expiry
    dt = T / (time_steps - 1)

    i = -1
    cf[:, i] = derivative.intrinsic_value(rf_sim, i)
    V[:, i] = cf[:, i]

    i = -2
    X = rf_sim[:, i].ravel().reshape(-1, 1)
    Y = (V[:, i + 1].ravel().reshape(-1, 1) + derivative.cashflow(rf_sim, i).reshape(-1, 1))*np.exp(-rfr*dt)

    models = [Pipeline([('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(1)),
                        ('regressor', LinearRegression())]),
              Pipeline([('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(2)),
                        ('regressor', LinearRegression())]),
              Pipeline([('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(3)),
                        ('regressor', LinearRegression())]),
              Pipeline([('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(4)),
                        ('regressor', LinearRegression())]),
              Pipeline([('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(4)),
                        ('regressor', LassoCV())]),
              Pipeline([('scaler', StandardScaler()),
                        ('poly', PolynomialFeatures(2)),
                        ('regressor', PiecewiseRegressor(verbose=True,
                                                         binner=KBinsDiscretizer(n_bins=2)))])]
    # Value of continuation
    fig, ax = plt.subplots(2, 3, figsize=(8, 6))
    names = ['LinReg ord.1', 'LinReg ord.2', 'LinReg ord.3', 'LinReg ord.4',
             'Lasso ord.4', '2-piece LinReg ord.2']
    for axis, model, name in zip(ax.ravel(), models, names):
        model.fit(X=X, y=Y)
        axis.scatter(X, Y, marker='+')
        axis.plot(sorted(X), model.predict(sorted(X)).ravel(), color='crimson')
        axis.set_title(name)

    plt.tight_layout()
    plt.show()
    fig.savefig("003_Outputs/Regressions_Tsitsiklis.pdf")
    plt.close(fig)


if __name__ == "__main__":

    # plot_exposures()
    plot_epe()

    #plot_interest_rate_distribution()
    #plot_early_exercise()
    #plot_tilley_alpha_dependency()
    #plot_tsitsiklis_regressions()
