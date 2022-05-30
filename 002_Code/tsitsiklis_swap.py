import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def simulate_brownian_paths(r0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths:int):
    Z = np.random.normal(loc=0.0, scale=1.0, size=(n_paths, n_steps)).cumsum(axis=1)
    R0 = np.ones(shape=(n_paths, 1))
    Z = np.hstack((R0, Z))
    dt = T/n_steps
    t = np.linspace(start=0.0, stop=T, num=n_steps + 1)
    paths = r0*np.exp((mu - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z).T
    return pd.DataFrame(paths, index=t)


if __name__ == '__main__':

    n_paths = 1000
    n_steps = 100
    mu = 0.01
    sigma = 0.1
    r0 = 0.01
    T = 5.0
    N = 100000.
    swap_rate = 0.01

    gbm = simulate_brownian_paths(r0=r0, sigma=sigma, mu=mu, T=T,
                                  n_steps=n_steps, n_paths=n_paths)
    running_quantiles = gbm.quantile(q=[0.025, 0.99], axis=1).T

    gbm.plot(color='green', alpha=0.1, legend=False)
    running_quantiles.plot()
    plt.title(f'GBM interest rate:\n {mu=}, {sigma=}, {r0=}')
    plt.legend([])
    plt.ylabel("r(t)")
    plt.xlabel("t")
    plt.show()

    dt = int(n_steps/(5*4))
    payment_moments = [dt*i for i in range(1, 5*4 + 1)]
    float_rate = gbm.iloc[payment_moments, ]
    payments = (float_rate - swap_rate)*N
    traj_pnl = payments.sum(axis=0)

    sns.distplot(traj_pnl)
    plt.show()
    exp_val = traj_pnl.mean()


    rfs = {}
    swap_values = {}
    linregs = {}
    y_regressed = {}
    maxstep = payments.shape[0]

    for step in range(1, maxstep + 1):
        fig, ax = plt.subplots(1, 2)
        if step == 1:
            rfs[-step] = float_rate.iloc[-step, ]
            swap_values[-step] = 0.0*rfs[-step]
            linregs[-step] = LinearRegression()
            linregs[-step].fit(X=rfs[-step].values.reshape(-1, 1),
                               y=swap_values[-1].values.reshape(-1, 1))
            y_regressed[-step] = linregs[-step].predict(X=rfs[-step].values.reshape(-1, 1))
            ax[0].scatter(x=rfs[-step], y=swap_values[-1])
            ax[0].plot(y_regressed[-step], color = 'red')
            plt.title(f'Step: -{step}')
            plt.show()
        else:
            rfs[-step] = float_rate.iloc[-step, ]
            swap_values[-step] = payments.iloc[-step + 1:, ].sum(axis=0)
            linregs[-step] = LinearRegression()
            linregs[-step].fit(X=rfs[-step].values.reshape(-1, 1),
                               y=swap_values[-step].values.reshape(-1, 1))
            y_regressed[-step] = linregs[-step].predict(X=rfs[-step].values.reshape(-1, 1))
            res = pd.DataFrame.from_dict({'Regressed': y_regressed[-step].ravel(),
                                          'RF': rfs[-step].values,
                                          'Sum_CFs': swap_values[-step].values})
            ax[0].scatter(x=res['RF'], y=res['Sum_CFs'])
            ax[0].scatter(x=res['RF'], y=res['Regressed'], label ='LinReg')
            ax[1].hist(res['Regressed'])
            plt.show()

    fig, ax = plt.subplots()
    exps = [val.mean() for key, val in y_regressed.items()]
    EPE = [np.maximum(val, 0.0).mean() for key, val in y_regressed.items()]
    q_low = [np.quantile(val, 0.05/2) for key, val in y_regressed.items()]
    q_up = [np.quantile(val, 1 - 0.05/2) for key, val in y_regressed.items()]

    df = pd.DataFrame.from_dict({'Mean': exps,
                                 'Q_0.025': q_low,
                                 'Q_0.975': q_up,
                                 'EPE': EPE})
    df.plot()
    plt.show()