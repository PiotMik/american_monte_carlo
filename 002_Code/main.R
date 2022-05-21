setwd("D:/Classes/Mgr/Semestr_4/ImplementacjaModeliFinansowych/american_monte_carlo/002_Code")
set.seed(2022)
library(tidyverse)

source('InterestRateSimulation.R')
source('Tilley.R')
source('TsitsiklisVanRoy.R')

r0 <- 0.05
sigma <- 0.05
mu <- 0.002
T <- 1.0
nsteps <- 100
npaths <- 200

rt <- simulate_gbm(r0 = r0, sigma=sigma, mu=mu, T=T,
                   nsteps=nsteps, npaths=npaths)
t <- seq(0, T, T/nsteps)
matplot(t, rt, type='l')
