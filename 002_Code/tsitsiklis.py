import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from mlinsights.mlmodel import PiecewiseRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import seaborn as sns
sns.set()


def tsitsiklis_price(rfr, time_steps, derivative, rf_sim):

    V = np.zeros_like(rf_sim)
    cf = np.zeros_like(rf_sim)
    early_exercise_boundary = np.full_like(V[0, :], fill_value=np.nan)
    T = derivative.expiry
    dt = T / (time_steps - 1)

    for i in range(-1, -time_steps, -1):
        if i == -1:
            cf[:, i] = derivative.intrinsic_value(rf_sim, i)
            V[:, i] = cf[:, i]
        else:
            e = derivative.intrinsic_value(rf_sim, i)
            X = rf_sim[:, i].ravel().reshape(-1, 1)
            Y = (V[:, i + 1].ravel().reshape(-1, 1) + derivative.cashflow(rf_sim, i).reshape(-1, 1))*np.exp(-rfr*dt)

            regressor = PiecewiseRegressor(verbose=True,
                                           binner=KBinsDiscretizer(n_bins=2))

            model = Pipeline([('scaler', StandardScaler()),
                              ('poly', PolynomialFeatures(2)),
                              ('regressor', regressor)]).fit(X=X, y=Y)

            # Value of continuation
            c = model.predict(X=rf_sim[:, i].ravel().reshape(-1, 1)).ravel()

            if derivative.early_exercisable:
                V[:, i] = np.maximum(c, e)
                rf_at_ex = rf_sim[e >= c, i]
                if len(rf_at_ex):
                    early_exercise_boundary[i] = np.max(rf_at_ex) if derivative.option_type == "put" else np.min(rf_at_ex)
            else:
                V[:, i] = c

    V[:, 0] = V[:, 1].mean()*np.exp(-rfr*dt)

    return V[0, 0], V, early_exercise_boundary


if __name__ == '__main__':
    pass
