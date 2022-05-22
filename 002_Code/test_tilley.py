from tilley import *
import pytest


@pytest.mark.parametrize('input,result', [(np.array([0., 0., 0.]), 3),
                                          (np.array([0., 0., 1., 0.]), 4),
                                          (np.array([0., 0., 0., 1.]), 3),
                                          (np.array([0., 0., 1., 1., 0., 1.]), 2),
                                          (np.array([0., 0., 1., 1., 0., 1., 0., 0., 0.]), 9),
                                          (np.array([0., 0., 1., 1., 0., 0., 1., 0., 0.]), 9),
                                          (np.array([0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0.]), 6)])
def test_boundary(input, result):
    output = boundary(input)
    assert output == result


@pytest.mark.parametrize('seed,time_steps,result', [(123, 10, 6.56893950),
                                                    (2022, 30, 5.40187295),
                                                    (44, 50, 2.83294454)])
def test_tilley_price(seed, time_steps, result):
    Q, P = 10, 20
    opt = AmericanOption(option_type='call', strike=102, expiry=1.0)

    np.random.seed(seed)
    gbm_sim = gbm(r0=100., sigma=0.2, mu=0.01, steps=time_steps, paths=Q*P).T

    price = tilley_price(time_steps=time_steps, Q=Q, P=P,
                         derivative=opt, rf_sim=gbm_sim)
    assert np.round(price, 7) == np.round(result, 7)


if __name__ == '__main__':
    pass
