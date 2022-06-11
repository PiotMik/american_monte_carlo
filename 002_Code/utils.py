import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def gbm(r0=0.05, sigma=0.2, mu=0.01, T=1.0, steps=5, paths=2):
    dt = T/(steps - 1)
    Z = np.random.normal(size=(steps, paths))
    growth_fcts = np.exp((mu - sigma ** 2 / 2) * dt + Z * sigma * np.sqrt(dt))
    growth_fcts[0, :] = 1.0
    return r0 * growth_fcts.cumprod(axis=0)


def plot_distribution(V, quantiles, T, figax=None, saveas=None, epe=True):
    df_quantiles = pd.DataFrame(np.quantile(V, quantiles, axis=0).T,
                                columns=['$q_{' + f'{q:.3f}' + "}$" for q in quantiles],
                                index=np.linspace(0, T, V.shape[1]))
    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax

    df_quantiles.plot(ax=ax)

    if epe:
        df_quantiles['EPE'] = np.maximum(V, 0).mean(axis=0).T
        df_quantiles['EPE'].plot(ax=ax, linestyle="--", legend=True)

    plt.show()
    if saveas is not None:
        fig.savefig(f"003_Outputs/{saveas}")
        plt.close(fig)
    else:
        return fig, ax


def plot_early_exercise_boundary(V, T, ex_bound, quantiles, figax=None, saveas=None):

    df_quantiles = pd.DataFrame(np.quantile(V, quantiles, axis=0).T,
                                columns=[str(q) for q in quantiles],
                                index=np.linspace(0, T, V.shape[1]))
    df_quantiles['EPE'] = np.maximum(V, 0).mean(axis=0).T

    if figax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = figax
    eeb = pd.DataFrame(ex_bound, index=df_quantiles.index,
                       columns=['Early Exercise Boundary']).replace(0.0, np.nan)
    ax.plot(df_quantiles.index, V[np.random.choice(range(V.shape[0]), 1000), :].T,
            color='grey', alpha=0.2)
    eeb.plot(ax=ax)
    plt.show()
    if saveas is not None:
        fig.savefig(f"003_Outputs/{saveas}")
        plt.close(fig)
    else:
        return fig, ax


if __name__ == "__main__":
    pass
