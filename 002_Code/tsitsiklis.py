import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from derivatives import InterestRateSwap, AmericanOption
import seaborn as sns
from utils import gbm, plot_distribution
sns.set()

if __name__ == '__main__':

    rfr = 0.05
    sigma = 0.2
    r0 = 0.01
    T = 5.0

    paths = 1000
    time_steps = 21
    dt = T/(time_steps - 1)

    rf_sim = gbm(r0=r0, sigma=sigma, mu=rfr, steps=time_steps, paths=paths).T

    derivatives = [AmericanOption(notional=1_000, strike=0.01, expiry=T, option_type="put"),
                   InterestRateSwap(notional=1_000, fixed_rate=0.010501, expiry=T, position='receiver',
                                    payment_freq_pa=4.0)]
    derivative = derivatives[1]
    plt.plot(rf_sim.T)

    V = np.zeros_like(rf_sim)
    cf = np.zeros_like(rf_sim)
    disc = np.full_like(rf_sim, fill_value=np.exp(rfr * dt)).cumprod(axis=1)
    for i in range(-1, -time_steps, -1):
        if i == -1:
            cf[:, i] = derivative.intrinsic_value(rf_sim, i)
            V[:, i] = cf[:, i]
        else:
            disc_ = disc/disc[0, i]
            e = derivative.intrinsic_value(rf_sim, i)
            X = rf_sim[:, i].ravel().reshape(-1, 1)
            Y = V[:, i + 1].ravel().reshape(-1, 1)*np.exp(-rfr*dt)


            #regressor = LassoCV()
            regressor = PiecewiseRegressor(verbose=True,
                                           binner=KBinsDiscretizer(n_bins=2))

            model = Pipeline([('scaler', StandardScaler()),
                              ('poly', PolynomialFeatures(2)),
                              ('regressor', regressor)]).fit(X=X, y=Y)

            # Value of continuation
            c = model.predict(X=rf_sim[:, i].ravel().reshape(-1, 1)).ravel()

            if derivative.early_exercisable:
                V[:, i] = np.maximum(c, e)
            else:
                V[:, i] = c + derivative.cashflow(rf_sim, i)

            if True:
                fig, ax = plt.subplots()
                ax.scatter(X, Y)
                ax.scatter(X, model.predict(X).ravel(), color='red')
                plt.show()
    V[:, 0] = V[:, 1].mean()*np.exp(-rfr*dt)
    plot_distribution(V=V, quantiles=[0.025, 0.25, 0.5, 0.75, 0.975], T=derivative.expiry,
                      saveas=f"Tsitsiklis_Exposure_{type(derivative).__name__}.pdf")

    price = V[0, 0]

    d1 = (np.log(r0 / derivative.strike) + (rfr - sigma ** 2 / 2) * derivative.expiry) / (sigma * np.sqrt(derivative.expiry))
    d2 = d1 - sigma * np.sqrt(derivative.expiry)
    bs_price = norm.cdf(d1) * r0 - norm.cdf(d2) * derivative.strike * np.exp(-rfr * derivative.expiry)

    print(f"American Call Price: {price:.4f}")
    print(f"European Call Price: {bs_price:.4f}")
