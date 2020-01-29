#%%
import pandas as pd
import numpy as np
import pymc3 as pm
from matplotlib import pyplot as plt
from hospice_project.data.transformer import MyScaler
from hospice_project.models.bayes_vis import test_model
from sklearn.pipeline import Pipeline
import seaborn as sns

# Make a new prediction from the test set and compare to actual value
def test_model(trace, test_observation, dv_name,
               show_observed=False):
    # Print out the test observation data
    print('Test Observation:')
    print(test_observation)
    var_dict = {}
    for variable in trace.varnames:
        var_dict[variable] = trace[variable]

    # Results into a dataframe
    var_weights = pd.DataFrame(var_dict)

    # Standard deviation of the likelihood
    sd_value = var_weights['sd'].mean()

    # Actual Value
    actual = test_observation[dv_name]

    # Add in intercept term
    test_observation['Intercept'] = 1
    test_observation = test_observation.drop(dv_name)

    # Align weights and test observation
    var_weights = var_weights[test_observation.index]

    # Means for all the weights
    var_means = var_weights.mean(axis=0)

    # Location of mean for observation
    mean_loc = np.dot(var_means, test_observation)

    # Estimates of grade
    estimates = np.random.normal(loc=mean_loc, scale=sd_value,
                                 size=1000)

    # Plot all the estimates
    plt.figure(figsize=(8, 8))
    sns.distplot(estimates, hist=True, kde=True, bins=19,
                 hist_kws={'edgecolor': 'k', 'color': 'darkblue'},
                 kde_kws={'linewidth': 4},
                 label='Estimated Dist.')

    # Plot the actual grade
    if show_observed:
        plt.vlines(x=actual, ymin=0, ymax=5,
                   linestyles='--', colors='red',
                   label='True Grade',
                   linewidth=2.5)

    # Plot the mean estimate
    # plt.vlines(x=mean_loc, ymin=0, ymax=5,
    #            linestyles='-', colors='orange',
    #            label='Mean Estimate',
    #            linewidth=2.5)

    plt.legend(loc=1)
    plt.title('Density Plot for Test Observation');
    plt.xlabel(dv_name);
    plt.ylabel('Density');
    plt.show()

    # Prediction information
    if show_observed:
        print('True Value = %d' % actual)
    print('Average Estimate = %0.4f' % mean_loc)
    print('5%% Estimate = %0.4f    95%% Estimate = %0.4f' % (np.percentile(estimates, 5),
                                                             np.percentile(estimates, 95)))