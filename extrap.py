import numpy as np
import pandas as pd
import datetime

from adept import default_class_index, STOP_VALUE, MASK_VALUE
from keras.preprocessing import sequence


# from http://stackoverflow.com/questions/30330389/histogram-datetime-objects-in-numpy
to_timestamp = np.vectorize(
    lambda x: (pd.Timestamp(x).to_datetime()
               - datetime.datetime(1970, 1, 1)).total_seconds())
from_timestamp = np.vectorize(
    lambda x: np.datetime64(datetime.datetime.utcfromtimestamp(x)))


def fill_with_prev_nonzero(seq):
    output = [0] * len(seq)
    last_nonzero = 0
    for i, v in enumerate(seq):
        if seq[i]==0:
            output[i] = last_nonzero
        else:
            last_nonzero = output[i] = seq[i]
    return output


def extrap_bin(e_a, e_b, default_status='no',
               statuses=['no', 'asthma', 'remission', 'relapse'],
               aggregation='sum', granularity=30):
    """
    Extrapolate the value of both event sequences (e_a, e_b) for even reference
    evaluation timeframes specified in timeline. Returns time-modified 
    e_a and e_b events.

    Assumes e_b is a single-column DataFrame with pt_id+event_date MultiIndex.

    CURRENTLY: assumes that days are the smallest granularity of time
    """
    assert(isinstance(e_b, pd.Series) or e_b.shape[1]==1)
    to_event_name = np.vectorize(lambda x: statuses[int(x)])

    e_a_new, e_b_new = pd.DataFrame(), pd.DataFrame()
    start_bins, end_bins, num_bins, bins = [], [], [], []
    for pt_id in e_a.index.levels[0]:
        # print "calculating histogram for {}".format(pt_id)
        pt_e_a = e_a.loc[pt_id]
        num_feats = len(pt_e_a.columns)
        num_samples = pt_e_a.shape[0]

        if pt_id in e_b.index.levels[0]:
            start_bins.append(min(e_b.loc[pt_id].index.min(),
                                  pt_e_a.index.min()))
            end_bins.append(max(e_b.loc[pt_id].index.max(),
                                pt_e_a.index.max()))
        else:
            start_bins.append(pt_e_a.index.min())
            end_bins.append(pt_e_a.index.max())

        # adjust end-bins to make bins day-aligned
        end_bins[-1] = end_bins[-1] + datetime.timedelta(
            granularity - ((end_bins[-1] - start_bins[-1]).days % granularity))
        num_bins.append((end_bins[-1] - start_bins[-1]).days // granularity)
        
        x = to_timestamp(np.matlib.repmat(pt_e_a.index.values,
                                          1, num_feats)[0])
        y = np.array([[i] * num_samples
                      for i in range(num_feats)]).flatten()
        w = pt_e_a.values.flatten('F')

        hist2d, x_bins, y_bins = np.histogram2d(
            x, y, bins=[num_bins[-1], num_feats],
            range=[[to_timestamp(start_bins[-1]), to_timestamp(end_bins[-1])],
                   [0, num_feats]],
            weights=w)

        new_index = pd.MultiIndex.from_arrays([[pt_id] * (len(x_bins) - 1),
                                               from_timestamp(x_bins[:-1])])
        e_a_pt = pd.DataFrame(hist2d, index=new_index, columns=pt_e_a.columns)
        e_a_new = pd.concat([e_a_new, e_a_pt])

        if pt_id not in e_b.index.levels[0]:
            e_b_pt = pd.DataFrame([statuses[0]] * len(new_index),
                                  index=new_index, columns=['event_name'])
        else:
            # print " FYI found pt_id {} in the gold standard".format(pt_id)
            e_b_dates = e_b.loc[pt_id].index
            e_b_tmp = pd.DataFrame([0] * len(x_bins[:-1]),
                                   index=from_timestamp(x_bins[:-1]))
            for i, d in enumerate(to_timestamp(e_b_dates)):
                e_b_prevbin = from_timestamp(x_bins[x_bins <= d].max())
                e_b_prevbin.shape = (1,)
                if e_b_prevbin[0] not in e_b_tmp.index:
                    print "WARNING: Inserting bad index for pt_id={}".format(
                        pt_id)
                e_b_tmp.loc[e_b_prevbin[0]] = e_b.loc[pt_id].iloc[i].values[0]

            e_b_pt = pd.DataFrame(
                fill_with_prev_nonzero(e_b_tmp.values.flatten()),
                index=new_index, columns=['event_name'])
            
        e_b_pt = e_b_pt.applymap(lambda x: x if x != 0 else statuses[0])
        e_b_new = pd.concat([e_b_new, e_b_pt])
        
    return e_a_new, e_b_new


def extrap_begin(e_a, e_b, default_status='no'):
                 # statuses=['no', 'asthma', 'remission', 'relapse']):
    """
    Extrapolate the value of timeline B events (e_b) for the evaluation points
    specified in timeline A (e_a). Returns time-modified e_b events.

     > A multiple-status-per-patient (MSP) setting if e_a is a timeline of
       features per patient, and e_b is a timeline of statuses. (see join_msp)
     > A single-status-per-patient (SSP) in special conditions. (see join_ssp)

    Assumes e_b is a single-column DataFrame with pt_id+event_date MultiIndex.
    """
    assert(isinstance(e_b, pd.Series) or e_b.shape[1]==1)
    
    e_b_new = pd.DataFrame([default_status for i in e_a.index.values],
                           index=e_a.index, columns=e_b.columns).reset_index()
    for pt_id in e_a.index.levels[0]:
        if pt_id not in e_b.index.levels[0]:
            continue
        for date, event in e_b.loc[pt_id].itertuples():
            if not (event and date): # or event not in statuses: ## 
                continue

            e_b_new.loc[(e_b_new['pt_id'] == pt_id) &
                       (e_b_new['event_date'] >= date),
                        e_b.columns[0]] = event
    return e_b_new.set_index(['pt_id', 'event_date'])
    

def extrap_last(e_a, e_b, default_status=None):
    """
    Find the timeline B value corresponding final timeline A event.
    Returns those timeline B values at the appropriate timeline A times.

    Typically used as input to single-status-per-patient (SSP) aggregation.
    """

    # find the last date for each patient
    ssp_index = [(e[0], e[1]) for e in
                 e_a.reset_index('event_date')['event_date']
                 .max(level=0).reset_index().values]

    # create fake e_a data with the right pt_ids and event_dates
    e_a_dummy = pd.DataFrame(data=np.zeros(len(ssp_index)),
                             index=pd.MultiIndex.from_tuples(
                                 ssp_index, names=['pt_id', 'event_date']))
    # e_a_dummy = e_a.iloc[:,0].loc[ssp_index].reset_index(
    #     'event_date').drop_duplicates(subset='event_date', keep='last')
    return extrap_begin(e_a_dummy, e_b)


def join_msp(e_a, e_b):
    """
    Add a column with e_b to e_a, i.e., extrap_begin
    """
    return e_a.join(extrap_begin(e_a, e_b))


def join_ssp(e_a, e_b):
    """
    Add a column with e_b to e_a's last instance, i.e., extrap_last
    """
    ssp_index = [(e[0], e[1]) for e in
                 e_a.reset_index('event_date')['event_date']
                 .max(level=0).reset_index('pt_id').values]

    tmp = e_a.join(extrap_last(e_a, e_b), how='right')
    out = tmp.reset_index(['pt_id', 'event_date']).drop_duplicates(
        subset=['pt_id', 'event_date'], keep='last').drop(
            ['pt_id', 'event_date'], axis=1)
    out.index = pd.MultiIndex.from_tuples(
        ssp_index, names=['pt_id', 'event_date'])
    return out


###############################################################################
# MERGING -- before there was EXTRAPOLATING
###############################################################################
def merge_gold(ft, gs, default_status='no',
               # statuses=['no', 'asthma', 'remission', 'relapse']):
    statuses = ['no', 'hospitalization', 'discharge', 'death']):

    """
    NOTE: DEPRECATED for estimation, use extrap_begin; OK for aggregation
    Attempt to place the gold standard status {'no', 'asthma', 'remission',
    'relapse'} as the last column of the feature matrix.

    Implicitly decides that if there are dates in the feature matrix, we'll be
    interested in specifying the correct status by date --- a multiple-status-
    per-patient (MSP) setting.  If there are no dates in the feature matrix,
    we will decide that it is more like a single-status-per-patient (SSP)
    setting.
    """

    if isinstance(ft.index, pd.MultiIndex):
        # typically: MSP aggregated features or raw features
        ft['class'] = pd.Series([default_status for i in ft.index.values],
                                index=ft.index)
        ft_new = ft.reset_index()
        for pt_id in ft.index.levels[0]:
            # prev = None
            for event, date in gs.loc[pt_id]['event_date'].iteritems():
                if not (event and date) or event not in statuses:
                    continue

                ft_new.loc[(ft_new['pt_id'] == pt_id) &
                           (ft_new['event_date'] >= date),
                           'class'] = event
        return ft_new.set_index(['pt_id', 'event_date'])
    
    else:
        # typically: SSP aggregated features
        gs_mrg = gs.reset_index('event_name')['event_name']\
                   .fillna(default_status)
        gs_mrg[-gs_mrg.isin(statuses)] = default_status # '-' inverts

        ft_out = ft.join(gs_mrg)
        ft_out.rename(columns={'event_name': 'class'}, inplace=True)
        return ft_out


def merge_events(e_a, e_b, default_status='no',
               # statuses=['no', 'asthma', 'remission', 'relapse']):
    statuses = ['no', 'hospitalization', 'discharge', 'death']):

    """
    NOTE: DEPRECATED for estimation, use extrap_begin; OK for aggregation
    Attempt to place the gold standard status {'no', 'asthma', 'remission',
    'relapse'} as the last column of the feature matrix.

    Implicitly decides that if there are dates in the feature matrix, we'll be
    interested in specifying the correct status by date --- a multiple-status-
    per-patient (MSP) setting.  If there are no dates in the feature matrix,
    we will decide that it is more like a single-status-per-patient (SSP)
    setting.
    """

    if isinstance(e_a.index, pd.MultiIndex):
        # typically: MSP aggregated features or raw features
        e_a['class'] = pd.Series([default_status for i in e_a.index.values],
                                index=e_a.index)
        e_a_new = e_a.reset_index()
        for pt_id in e_a.index.levels[0]:
            # prev = None
            for event, date in e_b.loc[pt_id]['event_date'].iteritems():
                if not (event and date) or event not in statuses:
                    continue

                e_a_new.loc[(e_a_new['pt_id'] == pt_id) &
                           (e_a_new['event_date'] >= date),
                           'class'] = event
        return e_a_new.set_index(['pt_id', 'event_date'])
    
    else:
        # typically: SSP aggregated features
        e_b_mrg = e_b.reset_index('event_name')['event_name']\
                   .fillna(default_status)
        e_b_mrg[-e_b_mrg.isin(statuses)] = default_status # '-' inverts

        e_a_out = e_a.join(e_b_mrg)
        e_a_out.rename(columns={'event_name': 'class'}, inplace=True)
        return e_a_out


##### SHAPERS (i.e., add explicit timing or not)
def shape(method, **kwargs):
    if not method or method=="default":
        def reltime_default(ft, **extra):
            return shape_per_patient(ft, **kw)
        return reltime_default
    if method.lower()=="markov":
        def reltime_markov(ft, **extra):
            kw = kwargs.update(extra)
            if kw:
                return shape_add_markov(ft, **kw)
            else:
                return shape_add_markov(ft)
        return reltime_markov
    if method.lower()=="landmark_first":
        def reltime_landmark_first(ft, **extra):
            kw = kwargs.update(extra)
            if kw:
                return shape_landmark_first(ft, **kw)
            else:
                return shape_landmark_first(ft)
        return reltime_landmark_first
    if method.lower()=="landmark_own":
        def reltime_landmark_own(ft, **extra):
            kw = kwargs.update(extra)
            if kw:
                return shape_landmark_own(ft, **kwargs)
            else:
                return shape_landmark_own(ft)
        return reltime_landmark_own
    if method.lower()=="landmark_any":
        def reltime_landmark_any(ft, **extra):
            kw = kwargs.update(extra)
            if kw:
                return shape_landmark_any(ft, **kwargs)
            else:
                return shape_landmark_any(ft, **kwargs)
        return reltime_landmark_any

    
def shape_per_patient(ft, num_history=None, **kwargs):
    if num_history:
        print "WARNING: num_history={} on a shaper w/o history".format(
            num_history)
    return [ft.loc[pt] for pt in ft.index.levels[0]]


# def shape_add_markov(ft, num_history=3, scale=1./365., power=None, **kwargs):
#     for h in xrange(1, num_history + 1):
#         hs = str(h)
#         # create the time difference from h prev time slices
#         ft['markov' + hs] = (
#             ft.index.get_level_values(level=1).values -
#             np.roll(ft.index.get_level_values(level=1).values, h)).astype(
#                 'timedelta64[D]').astype(float)
#         if scale:
#             ft['markov' + hs] *= scale
#         if power:
#             ft['markov' + hs] **= power

#         # zero out first h values, accounting for pt with few docs
#         h_ft = ft.loc[:, 'markov' + hs]
#         for pt in ft.index.levels[0]:
#             to_zero = min(h, len(ft.loc[pt, 'markov' + hs]))
#             h_ft.loc[pt, :to_zero] = 0
#             ft.loc[:, 'markov' + hs] = h_ft
#     return [ft.loc[pt] for pt in ft.index.levels[0]]


def shape_add_markov(ft, num_history=3, scale=1./365., power=None, **kwargs):
    # print "markov timing with num_history={}, scale={}, power={}".format(
    #     num_history, scale, power)
    ft_3D = []
    for pt in ft.index.levels[0]:
        # TODO: Currently really slow. Change to np arrays?
        pt_ft = ft.loc[pt]
        for h in xrange(1, num_history + 1):
            hs = str(h)
            # create the time difference from h prev time slices
            pt_ft['markov' + hs] = (
                pt_ft.index.values -
                np.roll(pt_ft.index.values, h)).astype(
                    'timedelta64[D]').astype(float)
            if scale:
                pt_ft['markov' + hs] *= scale
            if power:
                pt_ft['markov' + hs] **= power
            # zero out first h values, accounting for pt with few docs
            to_zero = min(h, len(pt_ft['markov' + hs]))
            pt_ft['markov' + hs][:to_zero] = 0
        ft_3D.append(pt_ft)
    return ft_3D


def shape_landmark_first(ft, scale=1./365., power=None, **kwargs):
    '''shape_landmark_first: relative time from first reported time

    Options:
    : scale: Time differences are multipled by this power (default=1./365.)
    : power: Time differences are raised to this power (default=None=1.)
    '''
    ft_3D = []
    for pt in ft.index.levels[0]:
        pt_ft = ft.loc[pt]
        pt_ft['landmark-first'] = (
            pt_ft.index.values - pt_ft.index.values[0]
        ).astype('timedelta64[D]').astype('float')
        if scale:
            pt_ft['landmark-first'] *= scale
        if power:
            pt_ft['landmark-first'] **= power
        ft_3D.append(pt_ft)
    return ft_3D


def shape_landmark_own(ft, num_history=None, scale=1./365., power=None, **kwargs):
    '''shape_landmark_own: relative time from previous same-feat nonzero event(s)
    Returns a list of 2D numpy arrays, to be padded for use in DNNs

    Options:
    : scale: Time differences are multipled by this power (default=1./365.)
    : power: Time differences are raised to this power (default=None=1.)
    '''
    ft_3D = []
    for pt in ft.index.levels[0]:
        pt_ft = ft.loc[pt]
        nz_feats, nz_times = pt_ft.values.T.nonzero()
        ownfeat_diff = np.ones_like(pt_ft, dtype=float) * MASK_VALUE
        for nz_feat, nz_time in zip(nz_feats, nz_times):
            ownfeat_diff[nz_time + 1:, nz_feat] = (
                pt_ft.ix[nz_time+1:, nz_feat].index.values -
                pt_ft.index.values[nz_time]).astype(
                    'timedelta64[D]').astype('float')
        if scale:
            ownfeat_diff[ownfeat_diff!=MASK_VALUE] *= scale
        if power:
            ownfeat_diff[ownfeat_diff!=MASK_VALUE] **= power
        ft_3D.append(np.append(pt_ft.values, ownfeat_diff, axis=-1))
    return ft_3D


def shape_landmark_any(ft, num_history=3, scale=1./365., power=None, **kwargs):
    '''shape_landmark_any: relative time from prev nonzero, any feat
    Returns a list of 2D numpy arrays, to be padded for use in DNNs

    Options:
    : scale: Time differences are multipled by this power (default=1./365.)
    : power: Time differences are raised to this power (default=None=1.)
    '''
    ft_3D = []
    if num_history==None:
        num_history = 1
    for pt in ft.index.levels[0]:
        pt_ft = ft.loc[pt]
        nz_times, nz_feats = pt_ft.values.nonzero()
        anyfeat_diff = np.ones((pt_ft.shape[0], num_history),
                               dtype=float) * MASK_VALUE
        for nz_time in sorted(set(nz_times)):
            for i in range(num_history - 1, 0, -1):
                anyfeat_diff[nz_time + 1:, i] = anyfeat_diff[nz_time + 1:, i-1]
            anyfeat_diff[nz_time + 1:, 0] = (
                pt_ft.ix[nz_time + 1:].index.values -
                pt_ft.index.values[nz_time]).astype(
                    'timedelta64[D]').astype('float')
        if scale:
            anyfeat_diff[anyfeat_diff!=MASK_VALUE] *= scale
        if power:
            anyfeat_diff[anyfeat_diff!=MASK_VALUE] **= power
        pt_ft_rt = np.c_[pt_ft.values, anyfeat_diff]
        ft_3D.append(pt_ft_rt)
    return ft_3D



##### SYNCHRONIZATION STRATEGIES
def sync_last(ft_raw, gs_raw, class_index=None,
              shaper=shape_per_patient, num_history=None, scale=None, power=None):
    # if not class_index:
    #     class_index = physionet_class_index

    ft = ft_raw
    gs_unmapped = extrap_last(ft_raw, gs_raw)
    gs = gs_unmapped.iloc[:,0].apply(lambda x: class_index[x])

    ft_3D = shaper(ft, num_history=num_history, scale=scale, power=power)
    x = sequence.pad_sequences(ft_3D, value=MASK_VALUE)
    y = np.array(gs.values.tolist())
    return x, y


def sync_state(ft_raw, gs_raw, class_index=None):
    if not class_index:
        class_index = default_class_index

    ft = ft_raw
    gs_unmapped = extrap_begin(ft_raw, gs_raw)
    gs = gs_unmapped.iloc[:,0].apply(lambda x: class_index[x])
    
    ft_3D = [ft.loc[pt] for pt in ft.index.levels[0]]
    gs_3D = [gs.loc[pt].values.tolist() for pt in gs.index.levels[0]]
    x = sequence.pad_sequences(ft_3D, value=STOP_VALUE)
    y = sequence.pad_sequences(gs_3D, value=MASK_VALUE)
    return x, y


def sync_bin(ft_raw, gs_raw, granularity=30, class_index=None):
    if not class_index:
        class_index = default_class_index

    ft, gs_unmapped = extrap_bin(ft_raw, gs_raw, granularity=granularity)
    gs = gs_unmapped.iloc[:,0].apply(lambda x: class_index[x])
    
    ft_3D = [ft.loc[pt] for pt in ft.index.levels[0]]
    gs_3D = [gs.loc[pt].values.tolist() for pt in gs.index.levels[0]]
    x = sequence.pad_sequences(ft_3D, value=STOP_VALUE)
    y = sequence.pad_sequences(gs_3D, value=MASK_VALUE)
    return x, y

    
if __name__=="__main__":
    from time import time
    from adept import munge as mng
    from adept import extrap
    from adept import eval
    from adept import default_classes
    t_start = {}
    # ft = mng.read_events('srcdata/features_dev127ev.deid.csv', binarize=True)
    # gs = mng.read_events('srcdata/goldtiming_dev127ev.deid.csv')

    # pdfin_features = pdfin_aggregate(ft, force=False, train_corpus='dev127ev',
    #                                  tag='indexdate-binaryfeat', binarize=True)
    # mng.write_arff(extrap.join_ssp(ft, gs),
    #                output_file='gendata/dev127.unmodified.ssp.arff',
    #                relation="PdfInAsthmaClassification")
    # mng.write_arff(extrap.join_msp(ft, gs),
    #                output_file='gendata/dev127.unmodified.msp.arff',
    #                relation="PdfInAsthmaClassification")

    ### LOAD DATA
    train_corpus = 'r01b1997-2002' # 'ape35ev' # 
    tag = 'rnn00'
    force = False
    binarize = True
    N_FOLDS = 5
    N_EPOCHS = 20

    print "Loading data..."; t_start['load'] = time()
    ft_raw, gs_raw = mng.load_corpus(train_corpus, tag, binarize=binarize)
    gs_raw = gs_raw.applymap(lambda x: ''.join(i for i in x if not i.isdigit()))
    classes  = default_classes
    print "--- {}: {} s ---".format('load', (time() - t_start['load']))

