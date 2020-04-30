from sampling_utils import *
from collections import OrderedDict
import theano
import re
import pandas as pd
import datetime
import numpy as np
import scipy as sp
import pymc3 as pm
import patsy as pt
import theano.tensor as tt
# BUG: may throw an error for flat RVs
theano.config.compute_test_value = 'off'


class SpatioTemporalFeature(object):
    def __init__(self):
        self._call_ = np.frompyfunc(self.call, 2, 1)

    def __call__(self, times, locations):
        _times = [pd.Timestamp(d) for d in times]
        return self._call_(np.asarray(_times).reshape(
            (-1, 1)), np.asarray(locations).reshape((1, -1))).astype(np.float32)


class SpatioTemporalYearlyDemographicsFeature(SpatioTemporalFeature):
    """ TODO:
    * county data must be updated to include 2019/2020 demographic data
      |> fix call
    """

    def __init__(self, county_dict, group, scale=1.0):
        self.dict = {
            (year, county): val * scale
            for county, values in county_dict.items()
            for (g, year), val in values["demographics"].items()
            if g == group
        }
        super().__init__()

    def call(self, yearweekday, county):
        # TODO: do this properly when data is available!
        return self.dict.get((2018, county))
        # return self.dict.get((yearweekday.year,county))


class SpatialEastWestFeature(SpatioTemporalFeature):
    def __init__(self, county_dict):
        self.dict = {
            county: 1.0 if "east" in values["region"] else (
                0.5 if "berlin" in values["region"] else 0.0) for county,
            values in county_dict.items()}
        super().__init__()

    def call(self, yearweekday, county):
        return self.dict.get(county)


class TemporalFourierFeature(SpatioTemporalFeature):
    def __init__(self, i, t0, scale):
        self.t0 = t0
        self.scale = scale
        self.τ = (i // 2 + 1) * 2 * np.pi
        self.fun = np.sin if (i % 2) == 0 else np.cos
        super().__init__()

    def call(self, t, x):
        return self.fun((t - self.t0) / self.scale * self.τ)

class TemporalPeriodicPolynomialFeature(SpatioTemporalFeature):
    def __init__(self, t0, period, order):
        self.t0 = t0
        self.period = period
        self.order = order
        super().__init__()

    def call(self, t, x):
        tdelta = (t - self.t0).days % self.period
        return (tdelta / self.period) ** self.order


class TemporalSigmoidFeature(SpatioTemporalFeature):
    def __init__(self, t0, scale):
        self.t0 = t0
        self.scale = scale
        super().__init__()

    def call(self, t, x):
        t_delta = (t - self.t0) / self.scale
        return sp.special.expit(t_delta.days + (t_delta.seconds / (3600 * 24)))

class TemporalPolynomialFeature(SpatioTemporalFeature):
    def __init__(self, t0, tmax, order):
        self.t0 = t0
        self.order = order
        self.scale = (tmax - t0).days
        super().__init__()

    def call(self, t, x):
        t_delta = (t - self.t0).days / self.scale
        return t_delta ** self.order

class ReportDelayPolynomialFeature(SpatioTemporalFeature):
    def __init__(self, t0, t_max, order):
        self.t0 = t0
        self.order = order
        self.scale = (t_max - t0).days
        super().__init__()
    
    def call(self, t, x):
        _t = 0 if t <= self.t0 else (t - self.t0).days / self.scale
        return _t ** self.order

class IAEffectLoader(object):
    generates_stats = False

    def __init__(self, var, filenames, days, counties):
        self.vars = [var]
        self.samples = []
        for filename in filenames:
            try:
                with open(filename, "rb") as f:
                    tmp = pkl.load(f)
            except FileNotFoundError:
                print("Warning: File {} not found!".format(filename))
                pass
            except Exception as e:
                print(e)
            else:
                m = tmp["ia_effects"]
                ds = list(tmp["predicted day"])
                cs = list(tmp["predicted county"])
                d_idx = np.array([ds.index(d) for d in days]).reshape((-1, 1))
                c_idx = np.array([cs.index(c) for c in counties])
                self.samples.append(np.moveaxis(
                    m[d_idx, c_idx, :], -1, 0).reshape((m.shape[-1], -1)).T)

    def step(self, point):
        new = point.copy()
        # res = new[self.vars[0].name]
        new_res = self.samples[np.random.choice(len(self.samples))]
        new[self.vars[0].name] = new_res
        # random choice; but block structure <-- this must have "design matrix" shape/content
        return new

    def stop_tuning(self, *args):
        pass

    @property
    def vars_shape_dtype(self):
        shape_dtypes = {}
        for var in self.vars:
            dtype = np.dtype(var.dtype)
            shape = var.dshape
            shape_dtypes[var.name] = (shape, dtype)
        return shape_dtypes


class BaseModel(object):
    """
    Model for disease prediction.

    The model has 4 types of features (predictor variables):
    * temporal (functions of time)
    * spatial (functions of space, i.e. longitude, latitude)
    * county_specific (functions of time and space, i.e. longitude, latitude)
    * interaction effects (functions of distance in time and space relative to each datapoint)
    """

    def __init__(
            self,
            trange,
            counties,
            ia_effect_filenames,
            num_ia=16,
            model=None,
            include_ia=True,
            include_report_delay=True,
            include_demographics=True,
            include_temporal=True,
            include_periodic=True,
            orthogonalize=False):

        self.county_info = counties
        self.ia_effect_filenames = ia_effect_filenames
        self.num_ia = num_ia if include_ia else 0
        self.include_ia = include_ia
        self.include_report_delay = include_report_delay
        self.include_demographics = include_demographics
        self.include_temporal = include_temporal
        self.include_periodic = include_periodic
        self.trange = trange


        # hardcoded for testing: Start Ger; Lockdown Ger; First Restr. lifeted Ger
        ia_trend_switchpoints = [pd.Timestamp('2020-01-28'), 
                                 pd.Timestamp('2020-03-16'), 
                                 pd.Timestamp('2020-04-20')]

        self.features = {
            "temporal_trend": {
                "temporal_polynomial_{}".format(i): TemporalPolynomialFeature(
                    pd.Timestamp('2020-01-28'), pd.Timestamp('2020-03-30'), i)
                    for i in range(4)} if self.include_temporal else {},
            "temporal_seasonal": {
                "temporal_periodic_polynomial_{}".format(i): TemporalPeriodicPolynomialFeature(
                    pd.Timestamp('2020-01-28'), 7, i)
                    for i in range(4)} if self.include_periodic else {},
            "spatiotemporal": {
                "demographic_{}".format(group): SpatioTemporalYearlyDemographicsFeature(
                    self.county_info,
                    group) for group in [
                        "[0-5)",
                        "[5-20)",
                        "[20-65)"]} if self.include_demographics else {},
            "interaction_trend": {
                "ia_trend_sigmoid_{}".format(i): TemporalSigmoidFeature(
                    ia_trend_switchpoints[i], 2.0) for i in range(3)},
            "temporal_report_delay" : {
                 "report_delay": ReportDelayPolynomialFeature(
                     pd.Timestamp('2020-04-17'), pd.Timestamp('2020-04-22'), 4)}
                     if self.include_report_delay else {},
            "exposure": {
                        "exposure": SpatioTemporalYearlyDemographicsFeature(
                            self.county_info,
                            "total",
                            1.0 / 100000)}}

        self.Q = np.eye(self.num_ia, dtype=np.float32)
        if orthogonalize:
            # transformation to orthogonalize IA features
            T = np.linalg.inv(np.linalg.cholesky(
                gaussian_gram([6.25, 12.5, 25.0, 50.0]))).T
            for i in range(4):
                self.Q[i * 4:(i + 1) * 4, i * 4:(i + 1) * 4] = T

    def evaluate_features(self, days, counties):
        all_features = {}
        for group_name, features in self.features.items():
            group_features = {}
            for feature_name, feature in features.items():
                feature_matrix = feature(days, counties)
                group_features[feature_name] = pd.DataFrame(
                    feature_matrix[:, :], index=days, columns=counties).stack()
            all_features[group_name] = pd.DataFrame([], index=pd.MultiIndex.from_product(
                [days, counties]), columns=[]) if len(group_features) == 0 else pd.DataFrame(group_features)
        return all_features

    def init_model(self, target):
        days, counties = target.index, target.columns

        # extract features
        features = self.evaluate_features(days, counties)
        Y_obs = target.stack().values.astype(np.float32)

        I_T = features["interaction_trend"].values.astype(np.float32)

        T_S = features["temporal_seasonal"].values.astype(np.float32)
        T_T = features["temporal_trend"].values.astype(np.float32)
        T_D = features["temporal_report_delay"].values.astype(np.float32)
        TS = features["spatiotemporal"].values.astype(np.float32)

        log_exposure = np.log(
            features["exposure"].values.astype(np.float32).ravel())

        # extract dimensions
        num_obs = np.prod(target.shape)
        num_ia_t = I_T.shape[1]
        num_t_s = T_S.shape[1]
        num_t_t = T_T.shape[1]
        num_t_d = T_D.shape[1]
        num_ts = TS.shape[1]

        with pm.Model() as self.model:
            # interaction effects are generated externally -> flat prior
            IA = pm.Flat("IA", testval=np.ones(
                  (num_obs, self.num_ia)), shape=(num_obs, self.num_ia))

            # priors - neg binomial
            # δ = 1/√α
            δ = pm.HalfCauchy("δ", 10, testval=1.0)
            α = pm.Deterministic("α", np.float32(1.0) / δ)

            # time-varying ~~~ W_ia -> NUM_IA x 1 -> mu dims?
            # TODO: match up the dimensions
            W_ia_t = pm.Normal("W_t_ia", mu=0, sd=10, 
                               testval=np.zeros(num_ia_t), shape=num_ia_t)
            kappa = pm.Deterministic("kappa", tt.dot(I_T, W_ia_t))
            sigma_ia = pm.HalfCauchy("sigma_ia", 10, testval=1.0)

            W_ia = pm.Normal("W_ia", mu=kappa, sd=sigma_ia, 
                             testval=np.zeros(self.num_ia), shape=self.num_ia)

            # neg binomial model
            W_t_s = pm.Normal("W_t_s", mu=0, sd=10,
                              testval=np.zeros(num_t_s), shape=num_t_s)
            W_t_t = pm.Normal("W_t_t", mu=0, sd=10,
                              testval=np.zeros(num_t_t), shape=num_t_t)
            W_t_d = pm.Normal("W_t_d", mu=0, sd=10,
                              testval=np.zeros(num_t_d), shape=num_t_d)
            W_ts = pm.Normal("W_ts", mu=0, sd=10,
                             testval=np.zeros(num_ts), shape=num_ts)
            self.param_names = ["δ", "W_ia_t", "W_ia", "W_t_s", "W_t_t", "W_t_d", "W_ts"]
            self.params = [δ, W_ia_t, W_ia, W_t_s, W_t_t, W_t_d, W_ts]

            # calculate interaction effect 
            IA_ef = tt.dot(tt.dot(IA, self.Q), W_ia)

            # calculate mean rates
            μ = pm.Deterministic(
                "μ",
                tt.exp(
                    IA_ef +
                    tt.dot(T_S, W_t_s) + 
                    tt.dot(T_T, W_t_t) +
                    tt.dot(T_D, W_t_d) +
                    tt.dot(TS, W_ts) +
                    log_exposure))

            # constrain to observations
            pm.NegativeBinomial("Y", mu=μ, alpha=α, observed=Y_obs)

    def map_estimate():
        """ TODO Q: how to include IA?"""
        pass

    def sample_parameters(
            self,
            target,
            n_init=100,
            samples=1000,
            chains=None,
            cores=8,
            init="advi",
            target_accept=0.8,
            max_treedepth=10,
            **kwargs):
        """
            sample_parameters(target, samples=1000, cores=8, init="auto", **kwargs)

        Samples from the posterior parameter distribution, given a training dataset.
        The basis functions are designed to be causal, i.e. only data points strictly
        predating the predicted time points are used (this implies "one-step-ahead"-predictions).
        """
        # model = self.model(target)

        self.init_model(target)

        if chains is None:
            chains = max(2, cores)

        with self.model:
            # run!
            ia_effect_loader = IAEffectLoader(
                self.model.IA,
                self.ia_effect_filenames,
                target.index,
                target.columns)
            nuts = pm.step_methods.NUTS(
                vars=self.params,
                target_accept=target_accept,
                max_treedepth=max_treedepth)
            steps = [ia_effect_loader, nuts]
            trace = pm.sample(samples, steps, chains=chains, cores=cores,
                              compute_convergence_checks=False, **kwargs)
        return trace

    def sample_predictions(
            self,
            target_days,
            target_counties,
            parameters,
            init="auto"):
        # extract features
        features = self.evaluate_features(target_days, target_counties)

        T_S = features["temporal_seasonal"].values
        T_T = features["temporal_trend"].values
        T_D = features["temporal_report_delay"].values
        TS = features["spatiotemporal"].values
        log_exposure = np.log(features["exposure"].values.ravel())

        # extract coefficient samples
        α = parameters["α"]
        W_ia = parameters["W_ia"]
        W_t_s = parameters["W_t_s"]
        W_t_t = parameters["W_t_t"]
        W_t_d = parameters["W_t_d"]
        W_ts = parameters["W_ts"]

        ia_l = IAEffectLoader(None, self.ia_effect_filenames,
                              target_days, target_counties)

        num_predictions = len(target_days) * len(target_counties)
        num_parameter_samples = α.size
        y = np.zeros((num_parameter_samples, num_predictions), dtype=int)
        μ = np.zeros((num_parameter_samples, num_predictions),
                     dtype=np.float32)

        # only consider the mean effect of the delay polynomial // should be a function?!
        # mean_delay = np.zeros((num_predictions,))
        # for i in range(num_parameter_samples):
        #     mean_delay += np.dot(T_D, W_t_d[i])

        # mean_delay /= num_parameter_samples

        for i in range(num_parameter_samples):
            IA_ef = np.dot(
                np.dot(ia_l.samples[np.random.choice(len(ia_l.samples))], self.Q), W_ia[i])
            μ[i, :] = np.exp(IA_ef +
                             np.dot(T_S, W_t_s[i]) +
                             np.dot(T_T, W_t_t[i]) +
                             # mean_delay +
                             # np.dot(T_D, W_t_d[i]) + 
                             np.dot(TS, W_ts[i]) +
                             log_exposure)
            y[i, :] = pm.NegativeBinomial.dist(mu=μ[i, :], alpha=α[i]).random()

        return {"y": y, "μ": μ, "α": α}
