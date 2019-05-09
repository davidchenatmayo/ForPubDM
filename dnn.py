import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, GRU, LSTM, Activation
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout, Masking, Lambda
from keras.layers.merge import Concatenate, Multiply, Add
import datetime
import pickle
from imblearn.under_sampling import RandomUnderSampler

from adept import munge as mng
from adept import estimate as est
from adept import diff, extrap
from adept import MASK_VALUE, STOP_VALUE


def simple_classification():
    ### cross-val test
    print "STARTING CLASSIFICATION"
    print "Transforming data... sync-last..."; t_start['sync'] = time()
    x, y = extrap.sync_last(ft_raw, gs_raw, class_index=physionet_class_index)
    NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
    MAX_DOCS = x.shape[1]  # ape35:464 docs/pt; r01b:1225
    NUM_CLASSES = len(classes)

    print classes
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

    # # Define Model
    print "Defining model..."; t_start['model'] = time()
    model = Sequential()
    model.add(LSTM(32, input_shape=(MAX_DOCS, NUM_FEATS)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=METRICS)
    print "--- {}: {} s ---".format('model', (time() - t_start['model']))

    print "Evaluating model..."; t_start['eval'] = time()
    util.cross_validate(x, y, model, balancer=RandomUnderSampler,
                        n_folds=N_FOLDS, verbose=0, epochs=N_EPOCHS)
    print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))


def binary_simple_classification():
    ### cross-val test
    print "STARTING CLASSIFICATION"
    print "Transforming data... sync-last..."; t_start['sync'] = time()
    x, y = extrap.sync_last(ft_raw, gs_raw, class_index=physionet_class_index)
    NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
    MAX_DOCS = x.shape[1]  # ape35:464 docs/pt; r01b:1225
    NUM_CLASSES = len(classes)

    print classes
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

    # # Define Model
    print "Defining model..."; t_start['model'] = time()
    model = Sequential()
    model.add(LSTM(32, input_shape=(MAX_DOCS, NUM_FEATS)))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=METRICS)
    print "--- {}: {} s ---".format('model', (time() - t_start['model']))

    print "Evaluating model..."; t_start['eval'] = time()
    util.cross_validate_auc_2(x, y, model, balancer=None,
                        n_folds=N_FOLDS, verbose=0, epochs=N_EPOCHS)
    print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))

    
def vary_basic_rnn_unit_classification():
### sync-last sequence classification
    print "Transforming data... sync-last..."; t_start['sync'] = time()
    x, y = extrap.sync_last(ft_raw, gs_raw)
    NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
    MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
    NUM_CLASSES = len(classes)
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

    RNNs = [
        LSTM(16, input_shape=(MAX_DOCS, NUM_FEATS)),
        LSTM(32, input_shape=(MAX_DOCS, NUM_FEATS)),
        LSTM(64, input_shape=(MAX_DOCS, NUM_FEATS)),
        Bidirectional(LSTM(32), input_shape=(MAX_DOCS, NUM_FEATS)),
        Bidirectional(GRU(32), input_shape=(MAX_DOCS, NUM_FEATS))
    ]

    for RNN in RNNs:
        print "STARTING CLASSIFICATION, {}".format(str(RNN))
        # Define Model
        print "Defining model..."; t_start['model'] = time()
        model = Sequential()
        model.add(RNN)
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=METRICS)
        print "--- {}: {} s ---".format('model', (time() - t_start['model']))

        print "Evaluating model..."; t_start['eval'] = time()
        util.cross_validate(x, y, model, n_folds=N_FOLDS, epochs=N_EPOCHS, verbose=2)
        print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))


def test_models_classification():
    print "STARTING CLASSIFICATION -- different models"
    print "Transforming data... sync-last..."; t_start['sync'] = time()
    x, y = extrap.sync_last(ft_raw, gs_raw)
    NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
    MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
    NUM_CLASSES = len(classes)
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

    from keras.layers.embeddings import Embedding
    m = {}
    m['baseline'] = Sequential()
    m['baseline'].add(LSTM(32, input_shape=(MAX_DOCS, NUM_FEATS)))
    m['baseline'].add(Dense(NUM_CLASSES, activation='softmax'))
    m['baseline'].compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=METRICS)
    m['baselinedo'] = Sequential()
    m['baselinedo'].add(LSTM(32, input_shape=(MAX_DOCS, NUM_FEATS)))
    m['baselinedo'].add(Dropout(0.5))
    m['baselinedo'].add(Dense(NUM_CLASSES, activation='softmax'))
    m['baselinedo'].compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=METRICS)
    m['conv1d'] = Sequential()
    m['conv1d'].add(Conv1D(32, 4, activation='relu',
                           input_shape=(MAX_DOCS, NUM_FEATS)))
    m['conv1d'].add(LSTM(32))
    m['conv1d'].add(Dense(NUM_CLASSES, activation='softmax'))
    m['conv1d'].compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=METRICS)
    m['conv1ddo'] = Sequential()
    m['conv1ddo'].add(Conv1D(32, 4, activation='relu',
                           input_shape=(MAX_DOCS, NUM_FEATS)))
    m['conv1ddo'].add(LSTM(32))
    m['conv1ddo'].add(Dropout(0.5))
    m['conv1ddo'].add(Dense(NUM_CLASSES, activation='softmax'))
    m['conv1ddo'].compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=METRICS)


    for m_name, model in m.iteritems():
        print "Evaluating model... {}".format(m_name); t_start['eval'] = time()
        run_score = util.cross_validate(
            x, y, model, balancer=RandomUnderSampler,
            n_folds=N_FOLDS, verbose=0, epochs=N_EPOCHS)
        print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))
        scores[m_name] = run_score
    return scores
        

def classification(ft_raw, gs_raw, sync=extrap.sync_last, sync_test=None,
                   model=None, balancer=RandomUnderSampler, conflator=None,
                   n_folds=5, verbose=0, epochs=20):
    ### cross-val test
    print "STARTING CLASSIFICATION"
    print "Transforming data, {}".format(sync); t_start['sync'] = time()
    x, y = sync(ft_raw, gs_raw)
    NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
    MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
    if conflator:
        y = conflator(y)
    NUM_CLASSES = y.shape[-1]

    if sync_test:
        x_test, y_test = sync_test(ft_raw, gs_raw)
        MAX_TEST_DOCS = x_test.shape[1]
    else:
        x_test, y_test = x, y
        MAX_TEST_DOCS = MAX_DOCS
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

    # # Define Model
    if not model:
        print "Defining model..."; t_start['model'] = time()
        model = Sequential()
        model.add(LSTM(32, input_shape=(MAX_DOCS, NUM_FEATS)))
        model.add(Dropout(0.5))
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=METRICS)
        print "--- {}: {} s ---".format('model', (time() - t_start['model']))

    print "Evaluating model..."; t_start['eval'] = time()
    run_score = util.cross_validate(
        x, y, model, balancer=RandomUnderSampler,
        n_folds=n_folds, verbose=verbose, epochs=epochs)
    print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))
    return run_score


def sync_strategies_classification():
    from keras.preprocessing import sequence
    from collections import defaultdict, Counter
    from sklearn.model_selection import StratifiedKFold

    run_scores = {}
    for sync, sync_test in [# (extrap.sync_last, extrap.sync_last),
                            (extrap.sync_state, extrap.sync_last),
                            (extrap.sync_bin, extrap.sync_last)
                            ]:
        balancer = RandomUnderSampler
        n_folds = N_FOLDS
        epochs = N_EPOCHS
        batch_size = 256
        verbose = 0

        ### cross-val test
        print "STARTING CLASSIFICATION FOR MSP TRAIN, SSP TEST"
        print "Transforming data... {} v. {}".format(sync, sync_test)
        t_start['sync'] = time()
        x_sync1, y_sync1 = sync(ft_raw, gs_raw)
        NUM_FEATS = x_sync1.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
        MAX_DOCS = x_sync1.shape[1] # ape35:464 docs/pt; r01b:1225
        NUM_CLASSES = len(classes)

        x_sync2, y_sync2 = sync_test(ft_raw, gs_raw)
        MAX_TEST_DOCS = x_sync2.shape[1]

        x_sync1 = sequence.pad_sequences(
            x_sync1, value=MASK_VALUE,
            maxlen=max(MAX_TEST_DOCS, MAX_DOCS))
        if y_sync1.ndim == 3:
            y_sync1 = sequence.pad_sequences(
                y_sync1, value=MASK_VALUE,
                maxlen=max(MAX_TEST_DOCS, MAX_DOCS))
        x_sync2 = sequence.pad_sequences(
            x_sync2, value=MASK_VALUE,
            maxlen=max(MAX_TEST_DOCS, MAX_DOCS))
        if y_sync2.ndim == 3:
            y_sync2 = sequence.pad_sequences(
                y_sync2, value=MASK_VALUE,
                maxlen=max(MAX_TEST_DOCS, MAX_DOCS))
        
        print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

        # # Define Model
        print "Defining model..."; t_start['model'] = time()
        m_train = Sequential()
        m_train.add(LSTM(32, input_shape=(max(MAX_DOCS, MAX_TEST_DOCS), NUM_FEATS),
                         return_sequences=True))
        m_train.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
        m_train.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=METRICS)
        m_test = Sequential()
        m_test.add(LSTM(32, input_shape=(max(MAX_DOCS, MAX_TEST_DOCS), NUM_FEATS)))
        m_test.add(Dense(NUM_CLASSES, activation='softmax'))
        m_test.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=METRICS)
        print "--- {}: {} s ---".format('model', (time() - t_start['model']))

        print "Evaluating model... {}-fold cross-validation".format(n_folds)
        scores = defaultdict(list)
        count = {}
        init_weights = m_train.get_weights()

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for i, (train, test) in enumerate(skf.split(x_sync1, util.max_ind(y_sync1))):
            print " ... Fold {}".format(i)
            util.shuffle_weights(m_train, init_weights)

            # undersampling
            rebalancer = balancer(return_indices=True)
            _, y_res, idx = rebalancer.fit_sample(x_sync1[train].sum(axis=1),
                                                  util.max_ind(y_sync1[train]))
            x_train = x_sync1[train][idx]
            y_train = y_sync1[train][idx]

            history = m_train.fit(x_train, y_train, batch_size=batch_size,
                                  epochs=epochs, verbose=verbose)

            m_test.set_weights(m_train.get_weights())
            y_pred = m_test.predict(x_sync2[test], verbose=0)

            if y_pred.ndim == 3:
                # MSP setting: mask out non-existent docs
                MASK = x_sync1.sum(axis=2) != MASK_VALUE * x_sync1.shape[2]
                y_mgold = util.pick(y_sync2[MASK])
                y_mpred = util.pick(y_pred[MASK])
            else:
                # SSP setting
                y_mgold = util.pick(y_sync2[test])
                y_mpred = util.pick(y_pred)

            for metric in util.STD_METRICS:
                m_name = metric.func_name
                count[m_name] = metric(y_mgold, y_mpred)
                scores[m_name].append(count[m_name])
                print "{} {}: {}".format(
                    m_name, i, count[m_name])

        run_scores[sync.func_name + "|"
                   + sync_test.func_name] = util.nfold_means(
                       scores, metrics=util.STD_METRICS)
    return run_scores


# def cat_markov_reltime_classification():
#     ### sync-last sequence classification, duration since last in the input vector
#     compare_scores = {}
#     for hist in [3, 2, 1, 0]:
#         print "STARTING CLASSIFICATION markov {}".format(hist)
#         print "Transforming data... sync-last..."; t_start['sync'] = time()
#         x, y = extrap.sync_last(ft_raw, gs_raw,
#                                 shaper=extrap.shape_add_markov,
#                                 num_history=hist)
#         NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
#         MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
#         NUM_CLASSES = len(classes)
#         print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

#         # Define Model
#         print "Defining model..."; t_start['model'] = time()
#         model = Sequential()
#         model.add(LSTM(32, # output: seq of 32x1 vect
#                       input_shape=(MAX_DOCS, NUM_FEATS))) # add 'elapsed' feat
#         model.add(Dense(NUM_CLASSES, activation='softmax'))

#         model.compile(loss='categorical_crossentropy',
#                       optimizer='rmsprop',
#                       metrics=['accuracy',
#                                ]) # util.f1, util.precision, util.recall])
#         print "--- {}: {} s ---".format('model', (time() - t_start['model']))

#         print "Evaluating model..."; t_start['eval'] = time()
#         run_score = util.cross_validate(
#             x, y, model, n_folds=N_FOLDS, verbose=1,
#             init_weights=model.get_weights(), epochs=N_EPOCHS)
#         print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))
#         run_scores[hist] = run_score
#     return run_scores


def simple_tagging():
    # ### tagging
    print "STARTING TAGGING TASK"
    print "Transforming data... sync-state..."; t_start['sync'] = time()
    x, y = extrap.sync_state(ft_raw, gs_raw) # sync-state tagging
    # # print "Transforming data... sync-bin..."; t_start['sync'] = time()
    # # x, y = extrap.sync_bin(ft_raw, gs_raw) # sync-bin tagging
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))


    print "Defining model..."; t_start['model'] = time()
    NUM_FEATS = x.shape[2] # ape35:24 pos 24 neg; r01b:30 pos 30 neg
    MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
    NUM_CLASSES = len(classes)
    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(MAX_DOCS, NUM_FEATS)))
    # model.add(Embedding(24, input_shape=(MAX_DOCS, NUM_FEATS)))
    model.add(LSTM(32, return_sequences=True)) #, input_shape=(MAX_DOCS, NUM_FEATS)))
    model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=METRICS)
    print "--- {}: {} s ---".format('model', (time() - t_start['model']))

    print "Evaluating model..."; t_start['eval'] = time()
    run_score = util.cross_validate(
        x, y, model, n_folds=N_FOLDS, epochs=N_EPOCHS)
    print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))
    return scores
    

# def multiply_markov_reltime_classification():
#     ### sync-last sequence classification, duration since last in the input vector
#     scores = {}
#     for hist in [1,2,3,4,8,16]:
#         print "STARTING CLASSIFICATION markov {}, weighted, 1yr scale, sigmoid".format(hist)
#         print "Transforming data... sync-last..."; t_start['sync'] = time()
#         x, y = extrap.sync_last_markov(ft_raw, gs_raw, num_history=hist)
#         NUM_FEATS = ft_raw.shape[-1]
#         NUM_TIME_FEATS = x.shape[-1] - NUM_FEATS
#         MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
#         NUM_CLASSES = len(classes)
#         print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

#         # Define Model
#         print "Defining model..."; t_start['model'] = time()
#         x_ev = Input(shape=(MAX_DOCS, NUM_FEATS))
#         t_ev = Input(shape=(MAX_DOCS, NUM_TIME_FEATS))

#         # wt = Lambda(lambda x: x/365.)(t_ev) # scale by year
#         wt = Dense(1, activation='sigmoid')(wt) # turn into single weight
#         wt = Lambda(lambda x: K.repeat_elements(x, NUM_FEATS, 2))(wt)
#         xt = Multiply()([x_ev, wt])
#         h = LSTM(32)(xt)
#         y_1hot = Dense(NUM_CLASSES, activation='softmax', name='status')(h)

#         model = Model(inputs=[x_ev, t_ev], outputs=y_1hot)
#         model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
#                       metrics=METRICS)
#         print "--- {}: {} s ---".format('model', (time() - t_start['model']))

#         print "Evaluating model..."; t_start['eval'] = time()
#         run_score = util.cross_validate_multi_in(
#             [x[:,:,:NUM_FEATS], x[:,:,NUM_FEATS:]], y, model,
#             n_folds=N_FOLDS, verbose=1,
#             init_weights=model.get_weights(), epochs=N_EPOCHS)
#         scores['historygate'+str(hist)] = run_score
#         print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))

#     return scores


def binary_v_progression():
    run_scores = {}
    test_names = ['asthma_binary', 'asthma_progression']
    run_scores[test_names[0]] = classification(
        ft_raw, gs_raw, sync=extrap.sync_last, conflator=util.conflate_to_first_two,
        epochs=N_EPOCHS, n_folds=N_FOLDS, verbose=0)
    run_scores[test_names[1]] = classification(
        ft_raw, gs_raw, sync=extrap.sync_last, conflator=None,
        epochs=N_EPOCHS, n_folds=N_FOLDS, verbose=0)
    # scores = dict(zip(['run' + str(d) for d in [1,2,3]],
    #                  [for test_name, run_score in run_scores]))
    print util.scores_to_latex(run_scores, filename='2017_06_05_14_27binvsprog.tex')


def cat_landmark_first_classification():
    scores = {}
    for p in [1., 1./.3]:
        print "STARTING CLASSIFICATION landmark-first {}, concatenated, 1yr scale, sigmoid".format(p)
        print "Transforming data... sync-last..."; t_start['sync'] = time()
        x, y = extrap.sync_last(
            ft_raw, gs_raw, shaper=extrap.shape_landmark_first, parameterization=p)
        NUM_FEATS = x.shape[2]
        NUM_TIME_FEATS = x.shape[-1] - NUM_FEATS
        MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
        NUM_CLASSES = len(classes)
        print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

        # Define Model
        print "Defining model..."; t_start['model'] = time()
        model = Sequential()
        model.add(LSTM(32, # output: seq of 32x1 vect
                      input_shape=(MAX_DOCS, NUM_FEATS))) # add 'elapsed' feat
        model.add(Dense(NUM_CLASSES, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy',
                               ]) # util.f1, util.precision, util.recall])
        print "--- {}: {} s ---".format('model', (time() - t_start['model']))

        print "Evaluating model..."; t_start['eval'] = time()
        run_score = util.cross_validate(
            x, y, model,
            n_folds=N_FOLDS, verbose=1,
            init_weights=model.get_weights(), epochs=N_EPOCHS)
        scores['landmark-first-cat'+str(p)] = run_score
        print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))

    print util.scores_to_latex(scores, filename='2017_06_06_12_52landmarkfirst.log')


def model_concat_reltime(max_docs, num_feats, num_time_feats,
                         num_classes, metrics=['accuracy']):
    x_ev = Input(shape=(max_docs, num_feats))
    t_ev = Input(shape=(max_docs, num_time_feats))

    wt = Activation('sigmoid')(t_ev)
    xt = Concatenate()([x_ev, wt])
    h = LSTM(32)(xt)
    y_1hot = Dense(num_classes, activation='softmax', name='status')(h)

    model = Model(inputs=[x_ev, t_ev], outputs=y_1hot)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=metrics)
    return model


def model_multiply_reltime(max_docs, num_feats, num_time_feats,
                           num_classes, metrics=['accuracy']):
    x_ev = Input(shape=(max_docs, num_feats))
    t_ev = Input(shape=(max_docs, num_time_feats))

    wt = Dense(num_feats, activation='sigmoid')(t_ev)
    xt = Multiply()([x_ev, wt])
    h = LSTM(32)(xt)
    y_1hot = Dense(num_classes, activation='softmax', name='status')(h)

    model = Model(inputs=[x_ev, t_ev], outputs=y_1hot)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                  metrics=metrics)
    return model


def reltime_classification(ft_raw, gs_raw, sync=extrap.sync_last,
                           model_func=None, balancer=RandomUnderSampler,
                           conflator=None, filename=None, metrics=['accuracy'],
                           shaper=extrap.shape_landmark_own,
                           n_folds=10, verbose=0, epochs=20):
    scores = {}
    print "STARTING RELTIME CLASSIFICATION {}, {}".format(
        shaper.func_name, model_func.func_name)
    print "Transforming data... sync-last..."; t_start['sync'] = time()
    x, y = sync(ft_raw, gs_raw, shaper=shaper, class_index=physionet_class_index)
    NUM_DATA_FEATS = ft_raw.shape[-1]
    NUM_TIME_FEATS = x.shape[-1] - NUM_DATA_FEATS
    MAX_DOCS = x.shape[1] # ape35:464 docs/pt; r01b:1225
    NUM_CLASSES = len(classes)
    print "--- {}: {} s ---".format('sync', (time() - t_start['sync']))

    # Define Model
    print "Defining model..."; t_start['model'] = time()
    if not model_func:
        model_func = model_multiply_reltime
    model = model_func(MAX_DOCS, NUM_DATA_FEATS, NUM_TIME_FEATS,
                       NUM_CLASSES, metrics=metrics)
    print "--- {}: {} s ---".format('model', (time() - t_start['model']))

    print "Evaluating model..."; t_start['eval'] = time()
    run_score = util.cross_validate_multi_in(
        [x[:,:,:NUM_DATA_FEATS], x[:,:,NUM_DATA_FEATS:]], y, model,
        n_folds=n_folds, verbose=verbose, balancer=balancer,
        init_weights=model.get_weights(), epochs=epochs)
    print "--- {}: {} s ---".format('eval', (time() - t_start['eval']))

    if filename: # if there's a filename, this is a final test
        scores['{} {}'.format(shaper.func_name,
                              model_func.func_name)] = run_score
        print util.scores_to_latex(scores, filename=filename)
    return run_score


def run_landmark_classification():
    ctr = 0
    for shaper, model_func in [
            (extrap.shape('landmark_any', num_history=1), model_multiply_reltime),
            (extrap.shape('landmark_any', num_history=2), model_multiply_reltime),
            (extrap.shape('landmark_any', num_history=3), model_multiply_reltime),
            (extrap.shape('landmark_any', num_history=4), model_multiply_reltime),
            (extrap.shape_landmark_own, model_multiply_reltime),
            (extrap.shape('landmark_any', num_history=1), model_concat_reltime),
            (extrap.shape('landmark_any', num_history=2), model_concat_reltime),
            (extrap.shape('landmark_any', num_history=3), model_concat_reltime),
            (extrap.shape('landmark_any', num_history=4), model_concat_reltime),
            (extrap.shape_landmark_own, model_concat_reltime)
            ]:
        scores = {}
        run_score = reltime_classification(
            ft_raw, gs_raw, shaper=shaper,
            model_func=model_func,
            n_folds=10, epochs=50)
        scores[ctr + ' ' + shaper.func_name + ' ' + model_func.func_name] = run_score
        ctr += 1

    print util.scores_to_latex(scores, filename='2017_06_09_11_54reltime.log')


###############################################################################
if __name__ == '__main__':
    from time import time
    from adept import util
    from adept import default_classes, default_class_index
    from adept import physionet_classes, physionet_class_index
    t_start = {}

    ### LOAD DATA
    # train_corpus = 'r01b1997-2002' # 'ape35ev' #
    train_corpus = 'physionet2012' # 'ape35ev' #

    tag = 'rnn00'
    force = False
    binarize = False
    N_FOLDS = 10    # 10 is the value for experiments
    N_EPOCHS = 50   # 50 is the value for experiments
    METRICS = ['accuracy'] #, util.precision_kmetric,
               # util.recall_kmetric, util.f1_kmetric]

    print "binarize: " + str(binarize)
    print "Loading data..."; t_start['load'] = time()
    ft_raw, gs_raw = mng.load_corpus(train_corpus, tag, binarize=binarize)
    gs_raw = gs_raw.applymap(lambda x:''.join(i for i in x if not i.isdigit()))
    classes = physionet_classes
    class_index = physionet_class_index
    # classes = default_classes
    # class_index = default_class_index
    print "--- {}: {} s ---".format('load', (time() - t_start['load']))

    # simple_classification()
    scores = binary_simple_classification()
    # vary_basic_rnn_unit_classification()
    # simple_tagging()
    # classification(ft_raw, gs_raw, epochs=5, n_folds=5)
    # cat_markov_reltime_classification()
    # multiply_markov_reltime_classification()
    # run_landmark_classification()

    # for model_func, shaper_type, num_history in [
    #         (model_concat_reltime, 'landmark_any', h)
    #         for h in [1,2,3,4,6,8,12,16]]:
    #     # try:
    #     shaper = extrap.shape(shaper_type, num_history=num_history)
    #     scores = {}
    #     run_score = reltime_classification(
    #         ft_raw, gs_raw, shaper=shaper,
    #         model_func=model_func,
    #         n_folds=N_FOLDS, epochs=N_EPOCHS)
    #     scores[model_func.func_name + ' ' + shaper_type
    #            + ' ' + str(num_history)] = run_score
    #     # except:
    #     #     print "WARNING: failed test -- {} {} {}".format(
    #     #         model_func, shaper_type, num_history)

    # for model_func, shaper_type, num_history in [
    #         (model_concat_reltime, 'markov', h)
    #         for h in [1,2,3,4,6,8,12,16]]:
    #     shaper = extrap.shape(shaper_type, num_history=num_history)
    #     scores = {}
    #     run_score = reltime_classification(
    #         ft_raw, gs_raw, shaper=shaper,
    #         model_func=model_func,
    #         n_folds=N_FOLDS, epochs=N_EPOCHS)
    #     scores[model_func.func_name + ' ' + shaper_type
    #            + ' ' + str(num_history)] = run_score

    # for model_func, shaper_type in [
    #         (model_concat_reltime, 'landmark_own'),
    #         (model_multiply_reltime, 'landmark_own')]:
    #     try:
    #         shaper = extrap.shape(shaper_type)
    #         scores = {}
    #         run_score = reltime_classification(
    #             ft_raw, gs_raw, shaper=shaper,
    #             model_func=model_func,
    #             n_folds=N_FOLDS, epochs=N_EPOCHS)
    #         scores[model_func.func_name + ' ' + shaper_type
    #                + ' ' + str(num_history)] = run_score
    #     except:
    #         print "WARNING: failed test -- {} {} {}".format(
    #             model_func, shaper_type)
    
    print util.scores_to_latex(scores, filename='2017_09_28_20_31reltime.log')

