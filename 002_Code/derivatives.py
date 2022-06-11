from dataclasses import dataclass
import numpy as np


@dataclass
class AmericanOption:
    early_exercisable = True
    notional: float
    strike: float
    expiry: float
    option_type: str

    def intrinsic_value(self, underlying_sim: np.array, t: int):
        ST = underlying_sim[:, t]
        mult = 1.0 if self.option_type.lower() == "call" else -1.0
        return self.notional*np.maximum(mult * (ST - self.strike), 0)

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


if __name__ == "__main__":
    pass
