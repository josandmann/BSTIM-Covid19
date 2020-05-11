from shared_utils import *
from BaseModel import BaseModel
from config import *
import pymc3 as pm
import pickle as pkl
import pandas as pd
import os

# SGE_TASK_ID=4 python sample_posterior.py
i = int(os.environ["SGE_TASK_ID"])-1

#NOTE: for jureca, extend to the number of available cores (chains and cores!)
num_samples = 250
num_chains = 4
num_cores = num_chains

# model_complexity, disease = combinations[i]

disease = "covid19"
prediction_region = "germany"

use_ia, use_report_delay, use_demographics, trend_order, periodic_order = combinations[i]

# use_interactions, use_report_delay = combinations_ia_report[model_complexity]

filename_params = "../data/mcmc_samples_backup/parameters_swp_{}_{}".format(disease, i)
filename_pred = "../data/mcmc_samples_backup/predictions_swp_{}_{}.pkl".format(disease, i)
filename_model = "../data/mcmc_samples_backup/model_swp_{}_{}.pkl".format(disease, i)

# Load data
with open('../data/counties/counties.pkl', "rb") as f:
    county_info = pkl.load(f)

data = load_daily_data(disease, prediction_region, county_info)

first_day = data.index.min()
last_day = data.index.max()

data_train, target_train, data_test, target_test = split_data(
    data,
    train_start=first_day,
    test_start=last_day - pd.Timedelta(days=1),
    post_test=last_day + pd.Timedelta(days=1)
)

tspan = (target_train.index[0], target_train.index[-1])

print("training for {} in {} with model complexity {} from {} to {}\nWill create files {}, {} and {}".format(
    disease, prediction_region, i, *tspan, filename_params, filename_pred, filename_model))

model = BaseModel(tspan,
                  county_info,
                  ["../data/ia_effect_samples/{}_{}.pkl".format(disease,
                                                                i) for i in range(100)],
                  include_ia=use_ia,
                  include_report_delay=use_report_delay,
                  include_demographics=use_demographics,
                  trend_poly_order=trend_order,
                  periodic_poly_order=periodic_order)
"""
print("Sampling parameters on the training set.")
trace = model.sample_parameters(
    target_train,
    samples=num_samples,
    tune=100,
    target_accept=0.95,
    max_treedepth=15,
    chains=num_chains,
    cores=num_cores)

with open(filename_model, "wb") as f:
    pkl.dump(model.model, f)

with model.model:
    pm.save_trace(trace, filename_params, overwrite=True)
"""

"""
>>>> predictions with a hierarchical switchpoint are hard to get and not the goal?
>>>> Load traces and inspect after the fact!

print("Sampling predictions on the training set.")

pred = model.sample_predictions(target_train.index, target_train.columns, trace)
with open(filename_pred, 'wb') as f:
     pkl.dump(pred, f)
 """
