#distributions for random sample generation of interarrival times and service times

import numpy as np
import random

import numpy.random
from scipy.stats import truncnorm, norm
import scipy.special
import math

#all have support in the positive real numbers except normal and uniform
#normal distribution will be truncated into positive real support while parameters for uniform must be non-negative
distributions = ["exponential", "gamma", "beta", "chisquare", "uniform", "normal", "lognormal",
                 "weibull", "rayleigh"]

class dis(object): #with support as a subset of positive real numbers
    '''distribution with positive real support'''
    def __init__(self, dis_name = "exp", parameters = (1,)):
        dis_name = dis_name.lower()
        if type(parameters) in [int, float]: #only one parameter
            parameters = (parameters,)
        #1 Exponential
        if dis_name in ["exp","expon","exponential"]:
            self.name = "exponential"
            if len(parameters) != 1:
                raise Exception('incorrect number of parameters, a rate parameter should be provided')
            self.parameters = parameters
            lamb = parameters[0]
            if lamb <= 0:
                raise Exception('rate parameter must be positive')
            self.generate_function = lambda: np.random.exponential(1 / lamb, 1).item()
            self.mean = 1 / lamb
            self.var = 1 / (lamb**2)
        #2 Gamma
        elif dis_name in ["gamma"]:
            self.name = "gamma"
            if len(parameters) != 2:
                raise Exception('incorrect number of parameters, a shape parameter and a rate parameter should be provided')
            self.parameters = parameters
            alpha, beta = parameters #shape and rate
            if alpha <= 0 or beta <= 0:
                raise Exception('both parameters must be positive')
            self.generate_function = lambda: np.random.gamma(alpha, 1/beta, 1).item()
            self.mean = alpha / beta
            self.var = alpha / (beta**2)
        #3 Beta
        elif dis_name in ["beta"]:
            self.name = "beta"
            if len(parameters) != 2:
                raise Exception('incorrect number of parameters, two shape parameters should be provided')
            self.parameters = parameters
            alpha, beta = parameters #shapes
            if alpha <= 0 or beta <= 0:
                raise Exception('both parameters must be positive')
            self.generate_function = lambda: np.random.beta(alpha, beta, 1).item()
            self.mean = alpha / (alpha + beta)
            self.var = alpha * beta / ((alpha + beta)**2 * (alpha + beta + 1))
        #4 Chisquare
        elif dis_name in ["chisq", "chisquare", "chisquared"]:
            self.name = "chisquare"
            if len(parameters) != 1:
                raise Exception('incorrect number of parameters, one degree of freedom parameter should be provided')
            self.parameters = parameters
            k = parameters[0] #df
            self.generate_function = lambda: np.random.chisquare(k, 1).item()
            self.mean = k
            self.var = 2 * k
        #5 Uniform
        elif dis_name in ["unif", "uniform"]:
            self.name = "uniform"
            if len(parameters) != 2:
                raise Exception('incorrect number of parameters, two different boundary parameters should be provided')
            self.parameters = parameters
            a, b = parameters
            if a > b: #change so a < b
                a, b = b, a
            elif a == b:
                raise Exception('boundary parameters should be different')
            if a < 0:
                raise Exception('both parameters must be positive')
            self.generate_function = lambda: np.random.uniform(a, b, 1).item()
            self.mean = 1 / 2 * (a + b)
            self.var = 1 / 12 * (b - a)**2
        #6 Normal
        elif dis_name in ["norm", "normal"]: #truncated normal
            self.name = "normal (truncated)"
            if len(parameters) != 2:
                raise Exception('incorrect number of parameters, a mean and a standard deviation should be provided')
            self.parameters = parameters
            mu, sigma = parameters
            if sigma <= 0:
                raise Exception('standard deviation should be positive')
            if mu + 2*sigma < 0:
                raise Exception('mean parameter is too negative, it is more than two standard deviations away from 0')
            a, b = 0, np.inf #truncation boundary
            alpha, beta = (a - mu) / sigma, (b - mu) / sigma
            Z = norm.cdf(b, mu, sigma) - norm.cdf(a, mu, sigma)
            self.generate_function = lambda: truncnorm.rvs(alpha, beta, loc = mu, scale = sigma, size = 1).item()
            self.mean = mu + (norm.pdf(a, mu, sigma) - norm.pdf(b, mu, sigma)) / Z * sigma
            self.var = sigma**2 * (1 +
                                   ((alpha * norm.pdf(a, mu, sigma) - 0) / Z) -
                                   ((norm.pdf(a, mu, sigma) - norm.pdf(b, mu, sigma)) / Z)**2
                                   )
        #7 LogNormal
        elif dis_name in ["lognorm", "lognormal"]:
            self.name = "lognormal"
            if len(parameters) != 2:
                raise Exception('incorrect number of parameters, a shape parameter and a scale parameter should be provided')
            self.parameters = parameters
            mu, sigma = parameters
            if sigma <= 0:
                raise Exception('standard deviation should be positive')
            self.generate_function = lambda: np.random.lognormal(mu, sigma, 1).item()
            self.mean = math.exp(mu + sigma**2 / 2)
            self.var = (math.exp(sigma**2) - 1) * math.exp(2 * mu + sigma**2)
        #8 Weibull
        elif dis_name in ["weibull"]:
            self.name = "weibull"
            if len(parameters) != 2:
                raise Exception('incorrect number of parameters, a shape parameter and scale parameter should be provided')
            self.parameters = parameters
            k, lamb = parameters
            if k <= 0 or lamb <= 0:
                raise Exception('both parameters must be positive')
            self.generate_function = lambda: lamb * np.random.weibull(k, 1).item()
            self.mean = lamb * scipy.special.gamma(1 + 1/k)
            self.var = lamb**2 * (scipy.special.gamma(1 + 2/k) - scipy.special.gamma(1 + 1/k)**2)
        #9 Rayleigh
        elif dis_name in ["rayleigh"]:
            self.name = "rayleigh"
            if len(parameters) != 1:
                raise Exception('incorrect number of parameters, a scale parameter should be provided')
            self.parameters = parameters
            sigma = parameters[0]
            if sigma <= 0:
                raise Exception('scale parameter should be positive')
            self.generate_function = lambda: np.random.rayleigh(sigma, 1).item()
            self.mean = sigma * math.sqrt(math.pi / 2)
            self.var = sigma**2 * (4 - math.pi) / 2
        #
        else:
            raise Exception('incorrect distribution name')

    def __str__(self):
        return f"{self.name} {self.parameters} distribution"

    def generate_samples(self, n = 100, time_end = None, seed = 0):
        if seed != None: #set seed
            random.seed(seed)
            numpy.random.seed(seed)
        if time_end: #generate up to time_end
            arr = self.generate_function() #initialisation
            interarrivals, arrivals = [arr], [arr]
            while True:
                interarr = self.generate_function()
                arr += interarr
                if arr >= time_end:
                    break
                else:
                    interarrivals.append(interarr)
                    arrivals.append(arr)
            return tuple(interarrivals), tuple(arrivals)
        else: #generate n random variables
            interarrivals = tuple(self.generate_function() for _ in range(n))
            arrivals = [interarrivals[0]]
            for e in interarrivals[1:]:
                arrivals.append(arrivals[-1] + e)
            return interarrivals, tuple(arrivals)


