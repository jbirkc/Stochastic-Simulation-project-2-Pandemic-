# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 12:21:52 2023

@author: jbirk
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
#from scipy.stats import t
import math 
from scipy import stats

def simulate_seirs_model(N, I0, R0, E0,delta,  beta0, gamma, epsilon, sigma,num_days):
    S = np.zeros(num_days)
    E = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)
  

    S[0] = N - I0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    BETA = []
    for t in range(1, num_days):
        beta = beta0 *(1+  delta * math.cos(2* math.pi *t/365))
        new_infections = np.random.binomial(S[t-1], beta * I[t-1] / N)
        new_recoveries = np.random.binomial(I[t-1], gamma)

        S[t] = S[t-1] - new_infections + np.random.binomial(R[t-1], epsilon)
        E[t] = E[t-1] + new_infections - np.random.binomial(E[t-1], sigma)
        I[t] = I[t-1] + np.random.binomial(E[t-1], sigma) - new_recoveries
        R[t] = R[t-1] + new_recoveries- np.random.binomial(R[t-1], epsilon)
        BETA.append(beta*N)
        

    return S, E,I, R, BETA

def simulate_det_seirs_model(epsilon, delta, beta0,sigma, E0 , gamma, N, I0, R0, num_days):
    S = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)
    E = np.zeros(num_days)

    S[0] = N - I0 - R0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    for t in range(1, num_days):
        beta = beta0 *(1+  delta * math.cos(2* math.pi *t/365))
        new_infections = beta * S[t-1] * I[t-1]/N
        new_recoveries = gamma * I[t-1]
        new_infectious = E[t-1]*sigma
        
        S[t] = S[t-1] - new_infections + epsilon*R[t-1]
        E[t] = E[t-1] + new_infections - new_infectious
        I[t] = I[t-1] + new_infectious - new_recoveries
        R[t] = R[t-1] + new_recoveries - epsilon*R[t-1]

    return S,E, I, R
# Parameters
N = 1000  # Total population size
I0 = 5 # Initial number of infected individuals
E0 = 1 # initial number of exposed
R0 = 0  # Initial number of recovered individuals
beta0 = 0.25  # Infection rate
gamma = 1/18  # Recovery rate
sigma = 1/30
epsilon = 1/10
num_days = 1000
num_simulations = 100
delta = 0.8 # rate where the 



x = np.arange(1, num_days+1)

Sd, Ed, Id, Rd =  simulate_det_seirs_model(epsilon, delta, beta0,sigma, E0 , gamma, N, I0, R0, num_days)

# Arrays to store simulation results
all_S = np.zeros((num_simulations, num_days))
all_E = np.zeros((num_simulations, num_days))
all_I = np.zeros((num_simulations, num_days))
all_R = np.zeros((num_simulations, num_days))


# Run simulations
for i in range(num_simulations):
    # Simulate SIR model
    S, E, I, R, BETA =  simulate_seirs_model(N, I0, R0, E0,delta,  beta0, gamma, epsilon, sigma,num_days)
    all_S[i] = S
    all_E[i] = E
    all_I[i] = I
    all_R[i] = R
        
# Calculate mean and confidence interval
mean_S = np.mean(all_S, axis=0)
mean_E = np.mean(all_E, axis=0)
mean_I = np.mean(all_I, axis=0)
mean_R = np.mean(all_R, axis=0)

std_S = np.std(all_S, axis=0)
std_E = np.std(all_E, axis=0)
std_I = np.std(all_I, axis=0)
std_R = np.std(all_R, axis=0)

lower_ci_S = mean_S - 1.96 * std_S / np.sqrt(num_simulations)
lower_ci_E = mean_E - 1.96 * std_E / np.sqrt(num_simulations)
lower_ci_I = mean_I - 1.96 * std_I / np.sqrt(num_simulations)
lower_ci_R = mean_R - 1.96 * std_R / np.sqrt(num_simulations)

upper_ci_S = mean_S + 1.96 * std_S / np.sqrt(num_simulations)
upper_ci_E = mean_E+ 1.96 * std_E / np.sqrt(num_simulations)
upper_ci_I = mean_I + 1.96 * std_I / np.sqrt(num_simulations)
upper_ci_R = mean_R + 1.96 * std_R / np.sqrt(num_simulations)

# Plotting the mean and confidence interval
plt.plot(mean_S, label='Mean S', color='r', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round')
plt.plot(mean_E, label='Mean E', color='magenta', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round')
plt.plot(mean_I, label='Mean I', color='b', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round')
plt.plot(mean_R, label='Mean R', color='g', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round' )
plt.plot(BETA, label='beta', color='k', alpha = 1, linestyle = (0,(4,5)),
    dash_capstyle = 'round')
plt.fill_between(range(num_days), lower_ci_S, upper_ci_S, color='r', alpha=0.3)
plt.fill_between(range(num_days), lower_ci_E, upper_ci_E, color='magenta', alpha=0.3)
plt.fill_between(range(num_days), lower_ci_I, upper_ci_I, color='b', alpha=0.3)
plt.fill_between(range(num_days), lower_ci_R, upper_ci_R, color='g', alpha=0.3)

# Plotting the mean and confidence interval
#plt.plot(x, Sd, label='Susceptible', color='r')
#plt.plot(x, Ed, label='Exposed', color='y')
#plt.plot(x, Id, label='Infectious', color='b')
#plt.plot(x, Rd, label='Recovered', color='g')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SEIRS Model Simulation')
plt.legend()
plt.show()

