# -*- coding: utf-8 -*-
import itertools as it
import pickle as pkl
import os
from sampling_utils import *
from shared_utils import *

disease = "covid19"
nums_sample = range(100)
GID = int(os.environ["SGE_TASK_ID"])
num_sample = nums_sample[GID - 1]

filename = "../data/ia_effect_samples/{}_{}.pkl".format(disease, num_sample)

print("Running task {} - disease: {} - sample: {}\nWill create file {}".format(GID, disease, num_sample, filename))

with open('../data/counties/counties.pkl', "rb") as f:
    counties = pkl.load(f)

prediction_region = "germany"

indata = load_daily_data(disease, prediction_region, counties)
data = indata

rnd_tsel = np.random.Generator(np.random.PCG64())
times_by_day = uniform_times_by_day(data.index, rnd_tsel)

rnd_csel = np.random.Generator(np.random.PCG64())
locations_by_county=uniform_locations_by_county(counties, rnd_csel)

res = iaeffect_sampler(data, times_by_day, locations_by_county, temporal_bfs, spatial_bfs)
results = {"ia_effects": res, "predicted day": data.index,
            "predicted county": data.columns}

with open(filename, "wb") as file:
    pkl.dump(results, file)
