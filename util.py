import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit

from keras.callbacks import Callback, TensorBoard
import tensorflow as tf
from keras import backend as K

from adept import MASK_VALUE
from adept import STOP_VALUE


def pick(y):
    if len(y.shape) > 1:
        return np.argmax(y, axis=-1)
    elif not np.issubdtype(y.dtype, np.integer):
        out = y.round()
        out.dtype = int
        return out
    else:
        return y


def onehot_to_ind(y):
    return np.argmax(y, axis=-1)


def max_ind(y):
    output = y
    while len(output.shape) > 1:
        output = np.argmax(output, axis=-1)
    return output


def conflate_to_first_two(y):
    new_y = np.array([y[:,0], np.max(y[:,1:], axis=-1)]).T
    # new_classes = classes[:2]
    return new_y #, new_classes


def macro_f1(y_gold, y_pred):
    # return f1_score(y_gold, y_pred, average='macro')
    p = precision_score(y_gold, y_pred, average='macro')
    r = recall_score(y_gold, y_pred, average='macro')
    return 2 * p * r / (p + r)
    

def macro_precision(y_gold, y_pred):
    return precision_score(y_gold, y_pred, average='macro')
    

def macro_recall(y_gold, y_pred):
    return recall_score(y_gold, y_pred, average='macro')
    

def micro_prec(y_gold, y_pred):
    return precision_score(y_gold, y_pred, average='micro')


STD_METRICS = [macro_precision, macro_recall, macro_f1, micro_prec]


def scores_to_dataframe(scores):
    '''expect `scores` input like:
        {'test_name1': {'metric1': score, 'metric2':score},
         'test_name2': {'metric2': score, 'metric2':score}}
    '''
    df = pd.DataFrame.from_dict(scores, orient='index')
    try: # to get into preferred order
        cols = df.columns.tolist()
        for m in [x.func_name for x in reversed(STD_METRICS)]:
            cols.insert(0, cols.pop(cols.index(m)))
        df = df[cols]
    except:
        pass
    return df


def scores_to_latex(scores, filename=None):
    latex = scores_to_dataframe(scores).to_latex(float_format='%1.4f')
    if filename:
        with open(filename, 'wb') as f:
            f.write(latex)
    return latex


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
                
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print '\nTesting loss: {}, acc: {}\n'.format(loss, acc)

        
class SSPStatsCallback(Callback):
    '''Single Status per Patient (SSP) statistics -- classification
    '''
    def __init__(self, test_data):
        self.test_data = test_data
                
    def on_epoch_end(self, epoch, logs={}):
        count, scores = {}, defaultdict(list)
        x, y = self.test_data

        # loss, acc = self.model.evaluate(x, y, verbose=0)
        # print '\nTesting loss: {}, acc: {}'.format(loss, acc)

        y_pred = self.model.predict(x, verbose=0)

        for metric in STD_METRICS:
            m_name = metric.func_name
            count[m_name] = metric(pick(y), pick(y_pred))
            print "{}: {}".format(m_name, count[m_name])
        print '\n'


class AllStatsCallback(Callback):
    '''AllStatsCallback handles both the MSP and SSP settings
    Multiple Status per Patient (MSP) -- sequence tagging
    Single Status per Patient (SSP) -- patient classification
    '''
    
    def __init__(self, test_data):
        self.test_data = test_data
                
    def on_epoch_end(self, epoch, logs={}):
        count, scores = {}, defaultdict(list)
        x, y = self.test_data

        y_pred = self.model.predict(x, verbose=0)

        if y_pred.ndim == 3:
            # MSP setting: mask out non-existent docs
            MASK = x.sum(axis=2) != MASK_VALUE * x.shape[2]
            y_mgold = pick(y[MASK])
            y_mpred = pick(y_pred[MASK])
        else:
            # SSP setting
            y_mgold = pick(y)
            y_mpred = pick(y_pred)

        print '\n'
        print "Class balance:\n {}\n {}".format(
            Counter(y_mgold), Counter(y_mpred))
        for metric in STD_METRICS:
            m_name = metric.func_name
            count[m_name] = metric(y_mgold, y_mpred)
            print "{}: {}".format(
                m_name, count[m_name])


def single_class_precision(relevant_class):
    '''https://codingquestion.blogspot.com/2017/01/keras-custom-metric-for-single-class.html'''
    def fn(y_true, y_pred):
        class_id_true = K.argmax(y_true, axis=-1)
        class_id_preds = K.argmax(y_pred, axis=-1)
        # Replace class_id_true with class_id_preds for recall here
        acc_mask = K.cast(K.equal(class_id_true, relevant_class), 'int32')
        class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds),
                                  'int32') * acc_mask
        class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(acc_mask), 1)
        return class_acc
    return fn


def precision_kmetric(y_gold, y_pred):
    '''precision: Calculates macro-average precision in batch
    - Ignores classes with 0 gold positives
    - Assumes a tensorflow backend
    '''
    # TODO: implement with CTC for the sync_state case
    num_classes = K.int_shape(y_pred)[-1]
    c_gold = K.argmax(y_gold, axis=-1)
    c_pred = K.argmax(y_pred, axis=-1)

    # repeat these tensors for each class, so we can calculate scores in batch
    rep_gold = K.tile(K.expand_dims(c_gold), (1, num_classes))
    rep_pred = K.tile(K.expand_dims(c_pred), (1, num_classes))
    per_c_ones = K.ones_like(K.expand_dims(c_pred))
    rep_c = K.concatenate([K.cast(per_c_ones * c, 'int64')
                           for c in range(num_classes)])
    
    i_gold = K.cast(K.equal(rep_gold, rep_c), K.floatx())
    i_pred = K.cast(K.equal(rep_pred, rep_c), K.floatx())
    num_c_in_pred = K.sum(i_pred, axis=0)
    precs = K.sum(i_gold * i_pred, axis=0) / num_c_in_pred
    mod_precs = K.gather(precs, tf.where(tf.logical_not(tf.is_nan(precs))))
    return K.mean(mod_precs)

    
def recall_kmetric(y_gold, y_pred):
    # TODO: implement with CTC for the sync_state case
    num_classes = K.int_shape(y_pred)[-1]
    c_gold = K.argmax(y_gold, axis=-1)
    c_pred = K.argmax(y_pred, axis=-1)

    # repeat these tensors for each class, so we can calculate scores in batch
    rep_gold = K.tile(K.expand_dims(c_gold), (1, num_classes))
    rep_pred = K.tile(K.expand_dims(c_pred), (1, num_classes))
    per_c_ones = K.ones_like(K.expand_dims(c_pred))
    rep_c = K.concatenate([K.cast(per_c_ones * c, 'int64')
                           for c in range(num_classes)])
    
    i_gold = K.cast(K.equal(rep_gold, rep_c), K.floatx())
    i_pred = K.cast(K.equal(rep_pred, rep_c), K.floatx())
    num_c_in_gold = K.sum(i_gold, axis=0)
    recs = K.sum(i_gold * i_pred, axis=0) / num_c_in_gold
    mod_recs = K.gather(recs, tf.where(tf.logical_not(tf.is_nan(recs))))
    return K.mean(mod_recs)


def f1_kmetric(y_gold, y_pred):
    # TODO: implement with CTC for the sync_state case
    num_classes = K.int_shape(y_pred)[-1]
    c_gold = K.argmax(y_gold, axis=-1)
    c_pred = K.argmax(y_pred, axis=-1)

    # repeat these tensors for each class, so we can calculate scores in batch
    rep_gold = K.tile(K.expand_dims(c_gold), (1, num_classes))
    rep_pred = K.tile(K.expand_dims(c_pred), (1, num_classes))
    per_c_ones = K.ones_like(K.expand_dims(c_pred))
    rep_c = K.concatenate([K.cast(per_c_ones * c, 'int64')
                           for c in range(num_classes)])
    
    i_gold = K.cast(K.equal(rep_gold, rep_c), K.floatx())
    i_pred = K.cast(K.equal(rep_pred, rep_c), K.floatx())
    num_c_in_gold = K.sum(i_gold, axis=0)
    num_c_in_pred = K.sum(i_pred, axis=0)
    precs = K.sum(i_gold * i_pred, axis=0) / num_c_in_pred
    recs = K.sum(i_gold * i_pred, axis=0) / num_c_in_gold
    mod_precs = K.gather(precs, tf.where(tf.logical_not(tf.is_nan(precs))))
    mod_recs = K.gather(recs, tf.where(tf.logical_not(tf.is_nan(recs))))
    prec = K.mean(mod_precs)
    rec = K.mean(mod_recs)
    return 2 * prec * rec / (prec + rec)
        

# def mean_pred(y_true, y_pred):
#     '''Sample custom metric
#     '''
#     return K.mean(y_pred)

    
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    From [jkleint/shuffle_weights.py](https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668)
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def nfold_means(scores, metrics=STD_METRICS):
    xval_scores = {}
    print "Mean metrics across folds:"
    metrics.append('AUC')
    for metric in metrics:
        if metric == 'AUC':
            m_name = 'AUC'
        else:
            m_name = metric.func_name
        xval_scores[m_name] = sum(scores[m_name]
            ) / len(scores[m_name])
        
        print "{}: {}".format(m_name, xval_scores[m_name])


    return xval_scores


### EVALUATION STRATEGIES
def cross_validate(x, y, model, n_folds=10, init_weights=None, verbose=1,
                   batch_size=256, epochs=5, balancer=None,
                   metrics=STD_METRICS):
    '''Cross validation with `model`, weights get shuffled each fold
    '''

    print "Evaluating model... {}-fold cross-validation".format(n_folds)
    scores = defaultdict(list)
    count = {}
    if not balancer:
        from imblearn.under_sampling import RandomUnderSampler
        balancer = RandomUnderSampler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for i, (train, test) in enumerate(skf.split(x, max_ind(y))):
        print " ... Fold {}".format(i)
        shuffle_weights(model, init_weights)

        # undersampling
        rebalancer = balancer(return_indices=True)
        _, y_res, idx = rebalancer.fit_sample(x[train].sum(axis=1),
                                              max_ind(y[train]))
        x_train = x[train][idx]
        y_train = y[train][idx]

        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, verbose=verbose)

        y_pred = model.predict(x[test], verbose=0)

        if y_pred.ndim == 3:
            # MSP setting: mask out non-existent docs
            MASK = x[test].sum(axis=2) != MASK_VALUE * x[test].shape[2]
            y_mgold = pick(y[test][MASK])
            y_mpred = pick(y_pred[MASK])
        else:
            # SSP setting
            y_mgold = pick(y[test])
            y_mpred = pick(y_pred)

        for metric in metrics:
            m_name = metric.func_name
            count[m_name] = metric(y_mgold, y_mpred)
            scores[m_name].append(count[m_name])
            print "{} {}: {}".format(
                m_name, i, count[m_name])

    nfold_scores = nfold_means(scores, metrics=metrics)
    return nfold_scores


### EVALUATION STRATEGIES
def cross_validate_auc_2(x, y, model, n_folds=10, init_weights=None, verbose=1,
                   batch_size=256, epochs=5, balancer=None,
                   metrics=STD_METRICS):
    '''Cross validation with `model`, weights get shuffled each fold
    '''

    print "Evaluating model... {}-fold cross-validation".format(n_folds)
    scores = defaultdict(list)
    count = {}
    if not balancer:
        from imblearn.under_sampling import RandomUnderSampler
        balancer = RandomUnderSampler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for i, (train_index, test_index) in enumerate(skf.split(x, max_ind(y))):
        print " ... Fold {}".format(i)
        shuffle_weights(model, init_weights)

        # undersampling
        rebalancer = balancer(return_indices=True)
        _, y_res, idx = rebalancer.fit_sample(x[train_index].sum(axis=1),
                                              max_ind(y[train_index]))
        x_train = x[train_index][idx]
        y_train = y[train_index][idx]

        history = model.fit(x_train, y_train, batch_size=batch_size,
                            epochs=epochs, verbose=verbose)

        y_pred = model.predict(x[test_index], verbose=0)

        # print "shape of y_pred: " + str(y_pred.shape)

        y_mgold = pick(y[test_index])
        y_mpred = pick(y_pred)

        # print "shape of y_mgold: " + str(y_mgold.shape)
        # print "shape of y_mpred: " + str(y_mpred.shape)
        y_gold_auc = y[test_index, 3]
        y_pred_auc = y_pred[:, 3]
        # print y_gold_auc
        # print y_pred_auc


        for metric in metrics:
            m_name = metric.func_name
            count[m_name] = metric(y_mgold, y_mpred)
            scores[m_name].append(count[m_name])
            print "{} {}: {}".format(
                m_name, i, count[m_name])

        scores['auc'] = roc_auc_score(y_gold_auc, y_pred_auc)
        print "AUC: %.4f" % scores['auc']


    nfold_scores = nfold_means(scores, metrics=metrics)
    return nfold_scores




def cross_validate_multi_in(inputs, outputs, model,
                            n_folds=10, init_weights=None, verbose=1,
                            batch_size=256, epochs=5, balancer=None,
                            metrics=STD_METRICS):
    '''Cross validation, with multiple inputs possible in a list
    Only the first input is used for:
      - Sample selection for cross-validation for folds
      - Determinng which inputs should be masked
    '''
    # if isinstance(inputs, list) or isinstance(inputs, tuple): # use with regular xval
    x, y = inputs, outputs
    
    print "Evaluating model... {}-fold cross-validation".format(n_folds)
    scores = defaultdict(list)
    count = {}
    if not balancer:
        from imblearn.under_sampling import RandomUnderSampler
        balancer = RandomUnderSampler

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)
    for i, (train, test) in enumerate(skf.split(x[0], max_ind(y))):
        print " ... Fold {}".format(i)
        shuffle_weights(model, init_weights)

        # undersampling
        rebalancer = balancer(return_indices=True)
        _, y_res, idx = rebalancer.fit_sample(x[0][train].sum(axis=1),
                                              max_ind(y[train]))
        x_train = [xp[train][idx] for xp in x]
        y_train = y[train][idx]

        model.fit(x_train, y_train, batch_size=batch_size,
                  epochs=epochs, verbose=verbose)
        y_pred = model.predict([xp[test] for xp in x], verbose=0)


        if y_pred.ndim == 3:
            # MSP setting: mask out non-existent docs

            MASK = x[0][test].sum(axis=2) != MASK_VALUE * x[0][test].shape[2]
            y_mgold = pick(y[test][MASK])
            y_mpred = pick(y_pred[MASK])
        else:
            # SSP setting
            y_mgold = pick(y[test])
            y_mpred = pick(y_pred)

        for metric in metrics:
            m_name = metric.func_name
            count[m_name] = metric(y_mgold, y_mpred)
            scores[m_name].append(count[m_name])
            print "{} {}: {}".format(
                m_name, i, count[m_name])

    nfold_scores = nfold_means(
        scores, metrics=metrics)
    return nfold_scores


def loo_validate(x, y, model):
    '''LOO validation on all cases. This trains # models = # patients
    '''

    print "Evaluating model... LOO validation"
    # scores = []
    for i in xrange(1, len(x) + 1):
        print " ... LOO Iteration {}".format(i)
        shuffle_weights(model)
        model.fit(np.concatenate((x[:i], x[i+1:])),
                  np.concatenate((y[:i], y[i+1:])),
                  epochs=5, verbose=0, callbacks=[
                      AllStatsCallback((x[test], y[test]))
                  ])
    #     _, count = model.evaluate(x[i-1:i], y[i-1:i])
    #     scores.append(count)
    # print "LOO scores: {}/{}".format(sum(scores), len(scores))
    

def simple(x, y, model, test_size=0.25, init_weights=None, verbose=1,
                batch_size=256, epochs=5, metrics=STD_METRICS):
    '''Simple train-test-split validation with `model`
    Parameters
    ----------
    test_size : float in [0,1]; what proportion of data held out for testing
    '''

    print "Evaluating model... train-test split (test={})".format(test_size)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    history = model.fit(x_train, y_train, batch_size=batch_size,
                        epochs=epochs, verbose=verbose,
                        metrics=metrics
                        # , callbacks=[
                        #     AllStatsCallback((x_test, y_test))
                        # ]
    )


def rebalanced(x, y, model, test_size=0.25, init_weights=None, verbose=1,
               batch_size=256, epochs=5, balancer=None):
    '''Find the best balancer validation with `model`
    Parameters
    ----------
    test_size : float in [0,1]; what proportion of data held out for testing
    '''

    if not balancer:
        from imblearn.under_sampling import RandomUnderSampler
        balancer = RandomUnderSampler
    
    print "Evaluating model... train-test split (balancer={})".format(balancer)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    rebalancer = balancer(return_indices=True)
    _, y_res, idx = rebalancer.fit_sample(x_train.sum(axis=1), max_ind(y_train))
    x_train = x_train[idx]
    y_train = y_train[idx]
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, verbose=verbose, callbacks=[
                  AllStatsCallback((x_test, y_test))
              ])
