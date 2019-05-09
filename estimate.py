import numpy as np
from adept import munge as mng
import timeit

from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

class KdeModel:
    x_grid_params = (-2100, 1100)
    bandwidth = 90
    feature = [] # may be multivariate in the future
    event = []   # may be multivariate in the future

    def __init__(self, estimated_density, feature=['asthma'], event=[1],
                 N=None):
        self.density = estimated_density
        self.feature = feature
        self.event = event
        self.N = N

    def __getitem__(self, time):
        try:
            index = int(np.where(self.x_grid()==time)[0][0])
            return self.density[index]
        except IndexError:
            return 0.0

    def x_grid(self, start=None, end=None):
        if start and end:
            self.x_grid_params = (start, end)
            return np.array(range(start, end + 1))
        else:
            return np.array(range(self.x_grid_params[0],
                                  self.x_grid_params[1] + 1))


"""
The kde_* functions originate from a blog post by Jake Vanderplas on Pythonic
Perambulations.
https://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

Content is BSD licensed.
"""
def kde_scipy(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bandwidth / x.std(ddof=1), **kwargs)
    return kde.evaluate(x_grid)


def kde_statsmodels_u(x, x_grid, bandwidth=0.2, **kwargs):
    """Univariate Kernel Density Estimation with Statsmodels"""
    kde = KDEUnivariate(x)
    kde.fit(bw=bandwidth, **kwargs)
    return kde.evaluate(x_grid)
    
    
def kde_statsmodels_m(x, x_grid, bandwidth=0.2, **kwargs):
    """Multivariate Kernel Density Estimation with Statsmodels"""
    kde = KDEMultivariate(x, bw=bandwidth * np.ones_like(x),
                          var_type='c', **kwargs)
    return kde.pdf(x_grid)


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    if bandwidth in ['cv', 'cross_val', 'xval', 'xv']:
        from sklearn.grid_search import GridSearchCV
        grid = GridSearchCV(KernelDensity(),
                            {'bandwidth': np.linspace(0.1, 1.0, 30)},
                            cv=20) # 20-fold cross-validation
        grid.fit(x[:, None])
        print grid.best_params_

        kde = grid.best_estimator_
        return np.exp(kde.score_samples(x_grid[:, None]))
    else:
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)


kde_funcs = [kde_statsmodels_u, kde_statsmodels_m, kde_scipy, kde_sklearn]
kde_funcnames = ['Statsmodels-U', 'Statsmodels-M', 'Scipy', 'Scikit-learn']
functions = dict(zip(kde_funcnames, kde_funcs))
    

def pdfs_by_values(date_diff, x_grid=np.linspace(-1000, 2100, 10000),
                   bw=90, kde_func=kde_scipy, x_grid_params=None,
                   plot=False):
    """
    Calculate the KDE PDFs with a grid of feature/event value pairs
    """
    from matplotlib import pyplot as plt

    pdfs = {}
    if plot:
        event_list, feat_list = mng.events_and_feats(date_diff)
        fig, ax = plt.subplots(len(event_list), len(feat_list))
                           # , sharey=True)
        ax = ax.reshape(len(event_list), len(feat_list))

    if x_grid_params:
        x_grid = np.linspace(*x_grid_params)
        
    for col_name in date_diff.keys():
        fpdf = {}
        for featval in date_diff[col_name].keys():
            for status in date_diff[col_name][featval].keys():
                fpdf[(featval, status)] = KdeModel(
                    kde_func(date_diff[col_name][featval][status],
                             x_grid, bandwidth=bw),
                    N=len(date_diff[col_name][featval][status]))
                if plot:
                    i = event_list.index(status)
                    j = feat_list.index(featval)
                    ax[i][j].plot(x_grid,
                                  fpdf[(featval, status)].density,
                                  label=col_name)
                    ax[i][j].set_title("Feat value={} and status={}".format(
                        featval, status))

                    # place the legend outside the plot
                    box = ax[i][j].get_position()
                    ax[i][j].set_position(
                        [box.x0, box.y0, box.width * 0.8, box.height],
                        which='original')
                    ax[i][j].legend(loc='center left', prop={'size':5},
                                    bbox_to_anchor=(1, 0.5))
        pdfs[col_name] = fpdf
    if plot:
        return fig
    else:
        return pdfs


def pdfs_by_bandwidth(date_diff, x_grid=np.linspace(-2100, 1000, 10000),
                      bandwidths=[30, 90, 180], kde_func=kde_scipy,
                      x_grid_params=None,
                      plot=False):
    """
    Calculate the KDE PDFs with a grid of bandwidth settings
    """
    from matplotlib import pyplot as plt

    pdfs = {}
    if plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(len(bandwidths), 1, sharey=True)

    if x_grid_params:
        x_grid = np.linspace(*x_grid_params)

    for col_name in date_diff.keys():
        if not 1 in date_diff[col_name].keys()\
           or not 1 in date_diff[col_name][1].keys():
            continue
        for i, bw in enumerate(bandwidths):
            pdfs[bw] = kde_func(date_diff[col_name][1][1],
                                 x_grid, bandwidth=bw)
            if plot:
                ax[i].plot(x_grid, pdfs[bw], label=col_name)
                ax[i].set_title("Bandwidth={}".format(bw))
                ax[i].legend(loc='upper left', prop={'size':4})
    if plot:
        return fig
    else:
        return pdfs


def get_pdfs_for_corpus(corpus_name, tag='', data_dir='gendata', force=False,
                        x_grid_params=(-1000, 2100, 10000), bw=90,
                        binarize=False, names=None, extrapolator=None):
    """
    get_pdfs_for_corpus - convenience wrapper around pdfs_by_values
      - assumes time-differences between features (e_a) and gold (e_b)
    """
    import os.path
    import pickle
    from adept import diff as diff

    file_path = "{}/{}{}.p".format(data_dir, corpus_name, tag)
    if os.path.isfile(file_path):
        if not force:
            print "Model exists, loading. Run with force=True to force."
            return pickle.load(open(file_path, 'rb'))

    ft, gs, date_diff = mng.load_corpus(corpus_name, tag, extrapolator)
    pdfs = pdfs_by_values(date_diff, x_grid_params=x_grid_params, bw=bw)
    pickle.dump(pdfs, open(file_path, 'wb'))
    return pdfs


def find_common_features(pdfs, lines_to_show=8, event_a=[1],
                         event_b=['asthma', 'remission', 'relapse']):
    import itertools
    feat_list = pdfs.keys()
    featval_list, status_list = zip(
        *[(featval, status) for col in pdfs.keys()
          for featval, status in pdfs[col].keys()])

    # find the best lines_to_show to display
    count = []
    output = {}
    for col_name in pdfs.keys():
        for featval, status in pdfs[col_name].keys():
            count.append((col_name, featval, status,
                          pdfs[col_name][(featval, status)].N))
    count_mtx = pd.DataFrame(count)
    for featval, status in itertools.product(event_a, event_b):
        tmp = count_mtx[count_mtx[1]==featval]
        output[(featval, status)] = tmp[tmp[2]==status].sort_values(
            3, ascending=False)[:lines_to_show][0].values.tolist()
    return output


def plot_pdfs(pdfs, lines_to_show=5, to_file=None,
              x_grid=np.linspace(-1000, 2100, 10000), x_grid_params=None,
              event_a=[1], event_b=['asthma', 'remission', 'relapse'], bw=90):
    """
    Calculate the KDE PDFs with a grid of feature/event value pairs
    """
    from matplotlib import pyplot as plt

    best_feats = find_common_features(
        pdfs, event_a=event_a, event_b=event_b, lines_to_show=lines_to_show)

    if x_grid_params:
        x_grid = np.linspace(*x_grid_params)
    fig, ax = plt.subplots(len(event_b), len(event_a))
    fig.subplots_adjust(hspace=0.6)
    plt.locator_params(axis='y',nbins=6)
    ax = ax.reshape(len(event_b), len(event_a))
    max_yticks = 3
    yloc = plt.MaxNLocator(max_yticks)

    for featval, status in best_feats.keys():
        b = event_b.index(status)
        a = event_a.index(featval)
        for feat in best_feats[(featval, status)]:
            ax[b][a].plot(x_grid,
                          pdfs[feat][(featval, status)].density,
                          label=feat)
            ax[b][a].set_title("Probability of {} at time t,".format(status)
                               # feat if featval else "no " + feat)
                               + " given FEATURE observed at t=0")

            # place the legend outside the plot
            box = ax[b][a].get_position()
            ax[b][a].set_position(
                [box.x0, box.y0, box.width * 0.8, box.height*0.9],
                which='original')
            ax[b][a].legend(loc='center left', prop={'size':6},
                            bbox_to_anchor=(1, 0.5))
            ax[b][a].yaxis.set_major_locator(yloc)

    if to_file:
        plt.savefig(to_file)
    else:
        # plt.show()
        return fig

    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from adept import diff
    from adept import extrap
    
    # gs = mng.read_gold_standard('srcdata/goldtiming_ape35.deid.csv')
    # ft = mng.read_features('srcdata/features_ape35.deid.csv',
    #                        binarize=True)

    # asthma_diff = diff.diff_asthma(ft, gs, statuses_per_patient='single',
    #                         min_examples=15)

    train_corpus = 'ape35ev'
    tag = 'practice_print'
    force = False
    binarize = True

    e_a = mng.read_events('srcdata/features_ape35ev.deid.csv',
                          binarize=True)
    e_b = mng.read_events('srcdata/goldtiming_ape35ev.deid.csv')

    pdfs = get_pdfs_for_corpus(train_corpus, tag=tag, force=force,
                               binarize=binarize)
    plot_pdfs(pdfs, lines_to_show=7, bw=90,
              to_file='gendata/default_plot_bw90.pdf')
    # # these objects should really be using multi-indexes
    # print "producing pdfs for 'normal' estimation"
    # t0a = timeit.default_timer()
    # date_diff_normal = diff.diff_timelines(e_a, e_b, min_examples=15)
    # print " dates diffed, {} elapsed".format(timeit.default_timer()-t0a)
    # fig1 = pdfs_by_values(date_diff_normal, plot=True, bw=180)
    # print " pdfs calculated, {} elapsed".format(timeit.default_timer()-t0a)
    # # plt.tight_layout()
    # plt.savefig('ape35ev_pdfs_normal.pdf')
    # print " pdfs printed, {} elapsed".format(timeit.default_timer()-t0a)

    # # print "producing pdfs for 'extrap' estimation"
    # t1a = timeit.default_timer()
    # e_b2 = extrap.extrap_begin(e_a, e_b)
    # print " extrap done, {} elapsed".format(timeit.default_timer()-t1a)
    # date_diff_extrap = diff.diff_timelines(e_a, e_b2, min_examples=15)
    # print " dates diffed, {} elapsed".format(timeit.default_timer()-t1a)
    # fig2 = pdfs_by_values(date_diff_extrap, plot=True, bw=180)
    # print " pdfs calculated, {} elapsed".format(timeit.default_timer()-t1a)
    # plt.savefig('ape35ev_pdfs_extrap.pdf')
    # print " pdfs printed, {} elapsed".format(timeit.default_timer()-t1a)

