import numpy as np
from collections import defaultdict
from functools import partial
import itertools

def _to_dates(series, status):
    return series[series==status].reset_index(level=1)['event_date']
    

def _diff_dates(s1, s2, drop_duplicates=False, index='pt_id'):
    s = s2.subtract(s1, level=0)
    if drop_duplicates:
        s = s.reset_index().drop_duplicates().set_index(index)
    diff_array = np.array(map(lambda x: x.days, s[s.notnull()]))
    return diff_array


def diff_asthma(ft, gs, statuses_per_patient='single', min_examples=5,
                drop_nan=True, verbose=False):
    """
    DEPRECATED use read_events and diff_timelines instead
    """
    # nonempty_patient_features = ft.any(level=0)
    if statuses_per_patient.startswith(('M', 'm')): # multiple
        raise NotImplementedError

    status_set = gs.index.levels[1].unique()
    # status_set = [s for s in status_set
    #               if s != 0] # drop 0 not necessary since indexing

    date_diff = defaultdict(
        partial(defaultdict, partial(defaultdict, lambda: np.arange(0))))
    for col_name, features in ft.iteritems():
        featval_set = features.unique()
        if len(featval_set)<=1:
            continue # trivial feature
        
        for status, featval in itertools.product(status_set, featval_set):
            dates = _to_dates(features, featval)
            # single status per patient gold standard
            # gs_date = _to_dates(gs.swaplevel(0, 1, axis=0).loc[status], status)
            gs_date = gs.swaplevel(0, 1, axis=0).loc[status]['event_date']
            diff_array = _diff_dates(dates, gs_date)
            if min_examples and diff_array.shape[0] >= min_examples:
                date_diff[col_name][featval][status] = diff_array
            elif verbose:
                print "Excluded {}, val={}, status={}; {} example(s)".format(
                    col_name, featval, status, diff_array.size)

    return date_diff


def diff_timelines(e_a, e_b, min_examples=5, drop_nan=True, verbose=False):
    """
    Supports the diff-ing of two or more event timelines.
    `e_a` used for features, or any 1+ event columns
    `e_b` used for gold standard, or any 1 event column

    This constrains the diff to calculate against one variable.
    """
    assert(e_b.shape[1] == 1)
    e_b_valset = e_b.iloc[:,0].unique()

    date_diff = defaultdict(
        partial(defaultdict, partial(defaultdict, lambda: np.arange(0))))
    for col_name, e_a_col in e_a.iteritems():
        e_a_valset = e_a_col.unique()
        if len(e_a_valset)<=1:
            continue # trivial feature
        
        for e_a_val, e_b_val in itertools.product(e_a_valset, e_b_valset):
            e_a_dates = _to_dates(e_a_col, e_a_val)
            e_b_dates = _to_dates(e_b.iloc[:,0], e_b_val)
            diff_array = _diff_dates(e_a_dates, e_b_dates)
            if min_examples and diff_array.shape[0] >= min_examples:
                date_diff[col_name][e_a_val][e_b_val] = diff_array
            elif verbose:
                print "Excluded {}, e_a_val={}, e_b_val={}; #ex={}".format(
                    col_name, e_a_val, e_b_val, diff_array.size)

    return date_diff


