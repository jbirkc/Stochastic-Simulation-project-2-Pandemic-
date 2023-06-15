
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
#from scipy.stats import t
import math 
from scipy import stats


def simulate_sir_model(N, I0, R0, beta, gamma, num_days):
    S = np.zeros(num_days)
    I = np.zeros(num_days)
    R = np.zeros(num_days)

    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    for t in range(1, num_days):
        new_infections = np.random.binomial(S[t-1], beta * I[t-1] / N)
        new_recoveries = np.random.binomial(I[t-1], gamma)

        S[t] = S[t-1] - new_infections
        I[t] = I[t-1] + new_infections - new_recoveries
        R[t] = R[t-1] + new_recoveries

    return S, I, R

def simulate_det_sir_model(beta, gamma, N, I0, R0, num_days):
    S, I, R = np.zeros(num_days), np.zeros(num_days), np.zeros(num_days)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    for t in range(1, num_days):
        new_infections = beta * S[t-1] * I[t-1]/N
        new_recoveries = gamma * I[t-1]
        
        S[t] = S[t-1] - new_infections
        I[t] = I[t-1] + new_infections - new_recoveries
        R[t] = R[t-1] + new_recoveries
    return S, I, R
# Parameters
N = 1000  # Total population size
I0 = 5 # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
beta = 0.25  # Infection rate
gamma = 1/18  # Recovery rate
num_days = 100
num_simulations = 30

x = np.arange(1, num_days+1)
S, I, R  = simulate_sir_model(N, I0, R0, beta, gamma, num_days)
Sd, Id, Rd = simulate_det_sir_model(beta, gamma, N, I0, R0, num_days)

# Arrays to store simulation results
all_S = np.zeros((num_simulations, num_days))
all_I = np.zeros((num_simulations, num_days))
all_R = np.zeros((num_simulations, num_days))


# Run simulations
for i in range(num_simulations):
    # Simulate SIR model
    S, I, R = simulate_sir_model(N, I0, R0, beta, gamma, num_days)
    all_S[i] = S
    all_I[i] = I
    all_R[i] = R
        



# Calculate mean and confidence interval
mean_S = np.mean(all_S, axis=0)
mean_I = np.mean(all_I, axis=0)
mean_R = np.mean(all_R, axis=0)

std_S = np.std(all_S, axis=0)
std_I = np.std(all_I, axis=0)
std_R = np.std(all_R, axis=0)

lower_ci_S = mean_S - 1.96 * std_S / np.sqrt(num_simulations)
lower_ci_I = mean_I - 1.96 * std_I / np.sqrt(num_simulations)
lower_ci_R = mean_R - 1.96 * std_R / np.sqrt(num_simulations)

upper_ci_S = mean_S + 1.96 * std_S / np.sqrt(num_simulations)
upper_ci_I = mean_I + 1.96 * std_I / np.sqrt(num_simulations)
upper_ci_R = mean_R + 1.96 * std_R / np.sqrt(num_simulations)

# Plotting the mean and confidence interval
plt.plot(mean_S, label='Mean Susceptible', color='r', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round')
plt.plot(mean_I, label='Mean Infected', color='b', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round')
plt.plot(mean_R, label='Mean Recovered', color='g', alpha = 0.5, linestyle = (0,(4,5)),
    dash_capstyle = 'round' )

plt.fill_between(range(num_days), lower_ci_S, upper_ci_S, color='r', alpha=0.3)
plt.fill_between(range(num_days), lower_ci_I, upper_ci_I, color='b', alpha=0.3)
plt.fill_between(range(num_days), lower_ci_R, upper_ci_R, color='g', alpha=0.3)

# Plotting the mean and confidence interval
plt.plot(x, Sd, label='Susceptible', color='r')
plt.plot(x, Id, label='Infected', color='b')
plt.plot(x, Rd, label='Recovered', color='g')
plt.xlabel('Days')
plt.ylabel('Population')
plt.title('SIR Model Simulation')
plt.legend()
plt.text(28, 920, f'Initial number of infected: {I0}', fontsize=10, color ='gray')
plt.text(28, 980, f'Number of simulations: {num_simulations}', fontsize=10, color ='gray')
plt.show()

# do the k-test 

import numpy as np
from scipy.stats import ks_2samp

# Example arrays
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([2, 4, 6, 8, 10])

# Perform Kolmogorov-Smirnov test
statistic, p_value = ks_2samp(mean_R, Rd)

# Output the results
print("Kolmogorov-Smirnov statistic:", statistic)
print("p-value:", p_value)
