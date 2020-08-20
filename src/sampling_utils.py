# -*- coding: utf-8 -*-
import pymc3 as pm
import theano
import theano.tensor as tt
import numpy as np
import pandas as pd
import isoweek
import pickle as pkl
import datetime
import time
from collections import OrderedDict
from matplotlib import pyplot as pp
from geo_utils import jacobian_sq
# BUG: may throw an error for flat RVs
theano.config.compute_test_value = 'off'


def uniform_times_by_week(weeks, n=500):
    """ Samples n random timepoints within a week, per week. converts times to datetime obj."""
    res = OrderedDict()
    for week in weeks:
        time_min = datetime.datetime.combine(
            isoweek.Week(*week).monday(), datetime.time.min)
        time_max = datetime.datetime.combine(
            isoweek.Week(*week).sunday(), datetime.time.max)
        res[week] = np.random.rand(n) * (time_max - time_min) + time_min
    return res


def uniform_times_by_day(days, n=500):
    """ Samples n random timepoints within a day, per day. converts pd.Timestamps to datetime obj."""
    res = OrderedDict()
    for day in days:
        time_min = datetime.datetime.combine(day, datetime.time.min)
        time_max = datetime.datetime.combine(day, datetime.time.max)
        res[day] = np.random.rand(n) * (time_max - time_min) + time_min
    return res


def uniform_locations_by_county(counties, n=500):
    res = OrderedDict()
    for (county_id, county) in counties.items():
        tp = county["testpoints"]
        if n == len(tp):
            res[county_id] = tp
        else:
            idx = np.random.choice(tp.shape[0], n, replace=n > len(tp))
            res[county_id] = tp[idx]
    return res


def _np_times_and_counties(times_by_day, locations_by_county):
    """ convert dicts to np.arrays for faster access """

    convert_t_pd = np.frompyfunc(pd.Timestamp, 1, 1)
    convert_t_flt = np.frompyfunc(datetime.datetime.timestamp, 1, 1)

    np_times_by_day = pd.DataFrame.from_dict(
        times_by_day, orient='index').to_numpy(
        dtype='datetime64')
    np_times_by_day = convert_t_pd(np_times_by_day)
    np_times_by_day = convert_t_flt(np_times_by_day)

    max_coords = 0
    for item in locations_by_county.items():
        max_coords = max(len(item[1]), max_coords)
    np_locations_by_county = np.empty(
        [len(locations_by_county.keys()), max_coords, 2], dtype='float64')
    for i, item in enumerate(locations_by_county.items()):
        np_locations_by_county[i][:] = item[1][1]

    return (np_times_by_day, np_locations_by_county)


def _allocate_samples(data, times_by_day, locations_by_county, idx):
    """ calculate fixed dataframe for time/space samples """

    day_offset = np.where(idx)[0][0]
    n_total = data.sum().sum()

    n_samples_county = np.array(data.values).flatten('F')

    day_ids = np.tile(np.arange(len(data.index)), len(data.columns))
    day_samples = [day_ids[i] for i, samples in enumerate(
        n_samples_county) for x in range(samples)]

    av_time_of_day = [len(times_by_day[d]) for d in data.index.tolist()]
    av_times_sample = [av_time_of_day[day_ids[i]] for i, samples in enumerate(
        n_samples_county) for x in range(samples)]

    county_ids = np.repeat(np.arange(len(data.columns)), len(data.index))
    county_samples = [county_ids[i]
                      for i, samples in enumerate(n_samples_county) for x in range(samples)]

    av_county_locs = [len(locations_by_county[c]) for c in data.columns]
    av_locs_sample = [av_county_locs[county_ids[i]]
                      for i, samples in enumerate(n_samples_county) for x in range(samples)]
    return (
        n_total,
        day_offset,
        day_samples,
        av_times_sample,
        county_samples,
        av_locs_sample)


def _sample_time_and_space_pred(
        n_days,
        n_counties,
        times_by_day,
        day_offset,
        num_tps,
        av_times_sample,
        locs_by_county,
        av_locs_sample,
        rng_time,
        rng_loc):

    n_total = n_days * n_counties * num_tps
    rnd_time_sample = np.floor(
        av_times_sample *
        rng_time.random(n_total)).astype('int32')

    t_all = [times_by_day[day_offset + i][rnd_time_sample[(i * n_counties + j) * num_tps + x]] 
             for i in range(n_days) for j in range(n_counties) for x in range(num_tps)]

    rnd_loc_sample = np.floor(av_locs_sample *
                              rng_loc.random((n_total,))).astype('int32')

    x_all = [locs_by_county[j][rnd_loc_sample[(i * n_counties + j) * num_tps + x]]
             for i in range(n_days) for j in range(n_counties) for x in range(num_tps)]

    return t_all, x_all


def _sample_time_and_space(
        n_counties,
        n_total,
        times_by_day,
        day_offset,
        day_samples,
        av_times_sample,
        locs_by_county,
        county_samples,
        av_locs_sample,
        rng_time,
        rng_loc):
    """ fast kernel for time/space samples """

    n_all = n_total * n_counties
    av_times_sample_all = np.tile(av_times_sample, n_counties)
    rnd_times_sample_all = np.floor(
        av_times_sample_all *
        rng_time.random(
            (n_all,))).astype('int32')

    # result 1: sampled ts
    t_all = [times_by_day[day_offset + day][rnd_times_sample_all[j * n_total + i]]
             for j in range(n_counties) for i, day in enumerate(day_samples)]

    av_locs_sample_all = np.tile(av_locs_sample, n_counties)
    rnd_locs_sample_all = np.floor(av_locs_sample_all * rng_loc.random((n_all,))).astype('int32')

    x_all = [locs_by_county[county][rnd_locs_sample_all[j * n_total + i]]
             for j in range(n_counties) for i, county in enumerate(county_samples)]

    return t_all, x_all


def sample_ia_effects(data, days, counties, ia_bfs, num_features, num_tps=10):

    rng_time_pred = np.random.Generator(np.random.PCG64(42))
    rng_loc_pred = np.random.Generator(np.random.PCG64(42))
    rng_time_dat = np.random.Generator(np.random.PCG64(42))
    rng_loc_dat = np.random.Generator(np.random.PCG64(42))

    pred_data = pd.DataFrame(num_tps, index=data.index, columns=data.columns)
    idx = np.empty([len(data.index)], dtype='bool')
    idx.fill(True)
    n_days = len(data.index)
    n_counties = len(data.columns)

    # to numpy
    (times_by_day, locations_by_county) = _np_times_and_counties(days, counties)

    (_,
     day_offset,
     _,
     av_times_sample,
     _,
     av_locs_sample) = _allocate_samples(pred_data,
                                         days,
                                         counties,
                                         idx)

    t_pred_all, x_pred_all = _sample_time_and_space_pred(n_days,
                                                         n_counties,
                                                         times_by_day,
                                                         day_offset,
                                                         num_tps,
                                                         av_times_sample,
                                                         locations_by_county,
                                                         av_locs_sample,
                                                         rng_time_pred,
                                                         rng_loc_pred)

    result = np.zeros( (n_days, n_counties, num_features), dtype=np.float32)
    for i, day in enumerate(data.index):

        idx = ((day - pd.Timedelta(days=5)) <= data.index) * (data.index < day)
        subtable = data.iloc[idx]
        n_cases = subtable.sum().sum()

        if subtable.size != 0 and n_cases != 0:
            (n_total,
                day_offset,
                day_samples,
                av_times_sample,
                county_samples,
                av_locs_sample) = _allocate_samples(subtable,
                                                    days,
                                                    counties,
                                                    idx)

            t_data_all, x_data_all = _sample_time_and_space(n_counties,
                                                            n_total,
                                                            times_by_day,
                                                            day_offset,
                                                            day_samples,
                                                            av_times_sample,
                                                            locations_by_county,
                                                            county_samples,
                                                            av_locs_sample,
                                                            rng_time_dat,
                                                            rng_loc_dat)

        for j, _ in enumerate(data.columns):

            if subtable.size != 0 and n_cases != 0:
                t_data = t_data_all[j * n_total:(j + 1) * n_total]
                x_data = x_data_all[j * n_total:(j + 1) * n_total]
            else:
                t_data = []
                x_data = np.empty((0,2)) # needs shape for theano.

            offset = (i * n_counties + j) * num_tps
            t_pred = t_pred_all[offset:offset + num_tps]
            x_pred = x_pred_all[offset:offset + num_tps]

            result[i, j, :] = ia_bfs(t_pred, x_pred, t_data, x_data)

    return result


def sample_time_and_space(data, times_by_day, locations_by_county, rng_time, rng_loc):
    n_total = data.sum().sum()
    t_all = np.empty((n_total,), dtype=object)
    x_all = np.empty((n_total, 2))

    i = 0
    for (county_id, series) in data.iteritems():
        for (day, n) in series.iteritems():
            # draw n random times
            times = times_by_day[day]
            # idx = np.random.choice(len(times), n)
            idx = np.floor( (n*[len(times)]) * rng_time.random((n,))).astype('int32')
            t_all[i:i + n] = times[idx]

            # draw n random locations
            locs = locations_by_county[county_id]
            # idx = np.random.choice(locs.shape[0], n)
            idx = np.floor( (n*[locs.shape[0]]) * rng_loc.random((n,))).astype('int32')
            x_all[i:i + n, :] = locs[idx, :]

            i += n

    return t_all, x_all


def gaussian_bf(dx, σ):
    σ = np.float32(σ)
    res = tt.zeros_like(dx)
    idx = (abs(dx) < np.float32(5) * σ)  # .nonzero()
    return tt.set_subtensor(res[idx], tt.exp(
        np.float32(-0.5 / (σ**2)) * (dx[idx])**2) / np.float32(np.sqrt(2 * np.pi * σ**2)))


def gaussian_gram(σ):
    return np.array([[np.power(2 * np.pi * (a**2 + b**2), -0.5)
                      for b in σ] for a in σ])


def bspline_bfs(x, knots, P):
    knots = knots.astype(np.float32)
    idx = ((x >= knots[0]) & (x < knots[-1]))  # .nonzero()
    xx = x[idx]

    N = {}
    for p in range(P + 1):
        for i in range(len(knots) - 1 - p):
            if p == 0:
                N[(i, p)] = tt.where((knots[i] <= xx)
                                     * (xx < knots[i + 1]), 1.0, 0.0)
            else:
                N[(i, p)] = (xx - knots[i]) / (knots[i + p] - knots[i]) * N[(i, p - 1)] + \
                    (knots[i + p + 1] - xx) / (knots[i + p + 1] - knots[i + 1]) * N[(i + 1, p - 1)]

    highest_level = []
    for i in range(len(knots) - 1 - P):
        res = tt.zeros_like(x)
        highest_level.append(tt.set_subtensor(res[idx], N[(i, P)]))
    return highest_level


def jacobian_sq(latitude, R=6365.902):
    """
        jacobian_sq(latitude) // TODO: reexport from geo_utils? // import to geo_utils?1

    Computes the "square root" (Cholesky factor) of the Jacobian of the cartesian projection from polar coordinates (in degrees longitude, latitude) onto cartesian coordinates (in km east/west, north/south) at a given latitude (the projection's Jacobian is invariante wrt. longitude).
    """
    return R * (np.pi / 180.0) * (abs(tt.cos(tt.deg2rad(latitude))) *
                                  np.array([[1.0, 0.0], [0.0, 0.0]]) + np.array([[0.0, 0.0], [0.0, 1.0]]))


def build_ia_bfs(temporal_bfs, spatial_bfs):
    x1 = tt.fmatrix("x1")
    t1 = tt.fvector("t1")
    # M = tt.fmatrix("M")
    x2 = tt.fmatrix("x2")
    t2 = tt.fvector("t2")

    lat = x1[:, 1].mean()
    M = jacobian_sq(lat)**2

    # (x1,t1) are the to-be-predicted points, (x2,t2) the historic cases

    # spatial distance btw. each points in x1 and x2 with gramian M
    dx = tt.sqrt((x1.dot(M) * x1).sum(axis=1).reshape((-1, 1)) + \
         (x2.dot(M) * x2).sum(axis=1).reshape((1, -1)) - \
         2 * x1.dot(M).dot(x2.T)) 

    # temporal distance btw. each times in t1 and t2
    dt = t1.reshape((-1, 1)) - t2.reshape((1, -1))

    ft = tt.stack(temporal_bfs(dt.reshape((-1,))), axis=0)
    fx = tt.stack(spatial_bfs(dx.reshape((-1,))), axis=0)

    # aggregate contributions of all cases
    contrib = ft.dot(fx.T).reshape((-1,)) / tt.cast(x1.shape[0], "float32")

    return theano.function([t1, x1, t2, x2], contrib,
                           allow_input_downcast=True)


class IAEffectSampler(object):
    """ switching to days """

    def __init__(
            self,
            data,
            times_by_day,
            locations_by_county,
            temporal_bfs,
            spatial_bfs,
            num_tps=10,
            time_horizon=5,
            verbose=True):

        self.ia_bfs = build_ia_bfs(temporal_bfs, spatial_bfs)
        self.times_by_day = times_by_day
        self.locations_by_county = locations_by_county
        self._to_timestamp = np.frompyfunc(datetime.datetime.timestamp, 1, 1)
        self.data = data
        self.num_tps = num_tps
        self.time_horizon = time_horizon
        self.num_features = len(temporal_bfs(tt.fmatrix(
            "tmp"))) * len(spatial_bfs(tt.fmatrix("tmp")))
        self.verbose = verbose

    def __call__(self, days, counties):

        rng_loc_pred = np.random.Generator(np.random.PCG64(42))
        rng_time_pred = np.random.Generator(np.random.PCG64(42))
        rng_time = np.random.Generator(np.random.PCG64(42))
        rng_loc = np.random.Generator(np.random.PCG64(42))

        res = np.zeros((len(days), len(counties),
                        self.num_features), dtype=np.float32)
        for i, day in enumerate(days):
            for j, county in enumerate(counties):
                idx = ((day - pd.Timedelta(days=5)) <=
                       self.data.index) * (self.data.index < day)
                # print("sampling day {} for county {} using data in range {}".format(day, county, idx))
                t_data, x_data = sample_time_and_space(self.data.iloc[idx],
                                                       self.times_by_day,
                                                       self.locations_by_county,
                                                       rng_time,
                                                       rng_loc)
                t_pred, x_pred = sample_time_and_space(pd.DataFrame(self.num_tps, index=[
                                                       day], columns=[county]),
                                                       self.times_by_day,
                                                       self.locations_by_county,
                                                       rng_time_pred,
                                                       rng_loc_pred)
                res[i, j, :] = self.ia_bfs(self._to_timestamp(
                    t_pred), x_pred, self._to_timestamp(t_data), x_data)
            frac = (i + 1) / len(days)
            if self.verbose:
                print("⎹" + "█" * int(np.floor(frac * 100)) + " ░▒▓█"[int(((frac * 100) % 1) * 5)] + " " * int(
                    np.ceil((1 - frac) * 100)) + "⎸ ({:.3}%)".format(100 * frac), end="\r", flush=True)
        return res
