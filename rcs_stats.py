import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

# Calculations for output RCS from given scene


# Should be more sophisticated, for now, combine each freq and average
# frequencies on x axis, thetas on y axis
def dim_reduction(rcs):
    #pca = PCA(n_components=5, svd_solver='full')
    #pca.fit(rcs)
    #return pca.singular_values_

    return np.mean(rcs, axis=1)


def basic_stats(rcs):
    rcs = dim_reduction(rcs)
    stat_dict = dict(
        mean=np.mean(rcs),
        median=np.median(rcs),
        stdev=np.std(rcs),
        var=np.var(rcs),
        median_abs_dev=stats.median_abs_deviation(rcs),
        quantile25=np.quantile(rcs, 0.25),
        quantile50=np.quantile(rcs, 0.50),
        quantile75=np.quantile(rcs, 0.75),
        iqr=stats.iqr(rcs)
    )
    return stat_dict


