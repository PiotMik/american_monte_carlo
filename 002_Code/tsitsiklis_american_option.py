import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


def simulate_brownian_paths(r0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths:int):
    Z = np.random.normal(loc=0.0, scale=1.0, size=(n_paths, n_steps))
    Z[:, 0] = 0.0
    Z = Z.cumsum(axis=1)
    dt = T/n_steps
    t = np.linspace(start=0.0, stop=T, num=n_steps)
    paths = r0*np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.sqrt(dt) * Z).T/np.exp((mu - sigma ** 2 / 2) * dt)
    return pd.DataFrame(paths, index=t)


if __name__ == '__main__':

    n_paths = 5000
    n_steps = 50
    sigma = 0.2
    rfr = 0.05
    S0 = 100.0
    T = 5.0
    strike = 110.0
    dt = T/n_steps
    option_type = 'call'
    mult = 1.0 if option_type == "call" else -1.0

    gbm = simulate_brownian_paths(r0=S0, mu=rfr, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths)

    h = np.maximum(gbm - strike, 0).values
    C = np.zeros_like(gbm)
    V = np.zeros_like(gbm)

    for i in range(-1, -n_steps, -1):
        if i == -1:
            V[i, :] = h[i, :]
        else:
            itm = h[i, :] > 0
            X = gbm.iloc[i, itm]
            Y = V[i + 1, itm]*np.exp(-rfr*dt)

            reg = LinearRegression().fit(X=X.ravel().reshape(-1, 1),
                                         y=Y.ravel().reshape(-1, 1))
            C[i, :] = reg.predict(X=gbm.iloc[i, :].ravel().reshape(-1, 1)).ravel()
            V[i, :] = np.maximum(h[i, :], C[i, :])
    V[0, :] = V[1, :].mean()

    d1 = (np.log(S0 / strike) + (rfr - sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    bs_price = norm.cdf(d1) * S0 - norm.cdf(d2) * strike * np.exp(-rfr * T)

    print(f"American Call Price: {V[0, 0]:.4f}")
    print(f"European Call Price: {bs_price:.4f}")
