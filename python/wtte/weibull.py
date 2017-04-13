import numpy as np
############################## Python weibull functions
def cdf(t, a, b):
    t = np.double(t)+1e-35
    return 1-np.exp(-np.power(t/a,b))

def hazard(t, a, b):
    t = np.double(t)+1e-35
    return (b/a)*np.power(t/a,b-1)

def pdf(t, a, b):
    t = np.double(t)+1e-35
    return (b/a)*np.power(t/a,b-1)*np.exp(-np.power(t/a,b))

def cmf(t, a, b):
    t = np.double(t)+1e-35
    return cdf(t+1, a, b)

def pmf(t, a, b):
    t = np.double(t)+1e-35
    return cdf(t+1.0, a, b)-cdf(t, a, b)

def mode(a, b):
    # Continuous mode.
    # TODO (mathematically) prove how close it is to discretized mode
    mode = a * np.power((b - 1.0) / b, 1.0 / b)
    mode[b <= 1.0] = 0.0
    return mode

def mean(a, b):
    # Continuous mean. at most 1 step below discretized mean 
    # E[T ] <= E[Td] + 1 true for positive distributions. 
    from scipy.special import gamma
    return a*gamma(1.0+1.0/b)

def quantiles(a, b, p):
    return a*np.power(-np.log(1.0-p),1.0/b)

def mean(a, b):
    # Continuous mean. Theoretically at most 1 step below discretized mean
    # E[T ] <= E[Td] + 1 true for positive distributions.
    from scipy.special import gamma
    return a * gamma(1.0 + 1.0 / b)

def continuous_logLik(t, a, b, u=1):
    # With equality instead of proportionality. 
    return u*np.log(pdf(t, a, b))+(1-u)*np.log(1.0-cdf(t, a, b))

def discrete_logLik(t, a, b, u=1):
    # With equality instead of proportionality. 
    return u*np.log(pmf(t, a, b))+(1-u)*np.log(1.0-cdf(t+1.0, a, b))

def cemean(t, a, b):
    # TODO this is not tested yet.
    # conditional excess mean
    # (conditional mean age at failure)
    # http://reliabilityanalyticstoolkit.appspot.com/conditional_distribution
    from scipy.special import gamma
    from scipy.special import gammainc
    # Regularized lower gamma
    print 'not tested'

    v = 1. + 1. / b
    gv = gamma(v)
    L = (t / a) ^ b
    cemean = a * gv * np.exp(L) * (1 - gammaic(v, t / a) / gv)

    return cemean

def cequantile(t, a, b, p):
    # TODO this is not tested yet.
    # conditional excess quantile
    print 'not tested'
    L = (t / a) ^ b

    quantile = a * (-np.log(1 - p) - L) ^ (1 / b)

    return quantile