import numpy as np
# Python weibull functions


def cumulative_hazard(t, a, b):
    t = np.double(t)
    return np.power(t / a, b)


def hazard(t, a, b):
    t = np.double(t)
    return (b / a) * np.power(t / a, b - 1)


def cdf(t, a, b):
    t = np.double(t)
    return 1 - np.exp(-np.power(t / a, b))


def pdf(t, a, b):
    t = np.double(t)
    return (b / a) * np.power(t / a, b - 1) * np.exp(-np.power(t / a, b))


def cmf(t, a, b):
    t = np.double(t) + 1e-35
    return cdf(t + 1, a, b)


def pmf(t, a, b):
    t = np.double(t) + 1e-35
    return cdf(t + 1.0, a, b) - cdf(t, a, b)


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
    return a * gamma(1.0 + 1.0 / b)


def quantiles(a, b, p):
    return a * np.power(-np.log(1.0 - p), 1.0 / b)


def mean(a, b):
    # Continuous mean. Theoretically at most 1 step below discretized mean
    # E[T ] <= E[Td] + 1 true for positive distributions.
    from scipy.special import gamma
    return a * gamma(1.0 + 1.0 / b)


def continuous_loglik(t, a, b, u=1, equality=False):
    if equality:
        # With equality instead of proportionality.
        loglik = u * np.log(pdf(t, a, b)) + (1 - u) * \
            np.log(1.0 - cdf(t, a, b))
    else:
        # commonly optimized over: proportional terms w.r.t alpha,beta
        loglik = u * loglik(hazard(t, a, b)) - \
            loglik(cumulative_hazard(t, a, b))

    return loglik


def discrete_loglik(t, a, b, u=1, equality=False):
    if equality:
        # With equality instead of proportionality.
        loglik = u * np.log(pmf(t, a, b)) + (1 - u) * \
            np.log(1.0 - cdf(t + 1.0, a, b))
    else:
        # commonly optimized over: proportional terms w.r.t alpha,beta
        hazard0 = cumulative_hazard(t, a, b)
        hazard1 = cumulative_hazard(t + 1., a, b)
        loglik = u * np.log(np.exp(hazard1 - hazard0) - 1.0) - hazard1

    return loglik

# Conditional excess


class conditional_excess():

    def pdf(t, s, a, b):
        t = np.double(t)
        return hazard(t + s, a, b) * np.exp(-cumulative_hazard(t + s) + cumulative_hazard(t))

    def cdf(t, s, a, b):
        t = np.double(t)
        return 1 - np.exp(-cumulative_hazard(t + s) + cumulative_hazard(t))

    def quantile(t, a, b, p):
        # TODO this is not tested yet.
        # tests:
        #    cequantile(0., a, b, p)==quantiles(a, b, p)
        #    cequantile(t, a, 1., p)==cequantile(0., a, 1., p)
        # conditional excess quantile
        # t+s : Pr(Y<t+s|y>t)=p

        print('not tested')
        L = np.power((t + .0) / a, b)

        quantile = a * np.power(-np.log(1. - p) - L, 1. / b)

        return quantile

    def mean(t, a, b):
        # TODO this is not tested yet.
        # tests:
        #    cemean(0., a, b)==mean(a, b, p)
        #    mean(t, a, 1., p)==mean(0., a, 1., p) == a
        # conditional excess mean
        # E[Y|y>t]
        # (conditional mean age at failure)
        # http://reliabilityanalyticstoolkit.appspot.com/conditional_distribution
        from scipy.special import gamma
        from scipy.special import gammainc
        # Regularized lower gamma
        print('not tested')

        v = 1. + 1. / b
        gv = gamma(v)
        L = np.power((t + .0) / a, b)
        cemean = a * gv * np.exp(L) * (1 - gammainc(v, t / a) / gv)

        return cemean
