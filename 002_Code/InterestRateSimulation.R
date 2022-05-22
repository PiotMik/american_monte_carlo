library(tidyverse)

simulate_gbm <- function(r0 = 0.05, sigma = 0.1, mu = 0.01,
                         T = 5.0, nsteps=100, npaths=4){
  # funkcja symulujaca stope procentowa Blackiem-Scholesem
  dt <- T/nsteps
  Z <- array(rnorm(n = nsteps*npaths, mean = 0, sd = 1),
             dim=c(nsteps, npaths))*sigma*sqrt(dt) + (mu - sigma^2/2)*dt
  Xt <- apply(Z, 2, cumsum)
  
  rt <- r0*exp(Xt)
  return(rbind(rep(r0, npaths), rt))
}
