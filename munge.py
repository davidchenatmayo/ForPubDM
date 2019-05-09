import numpy as np
import pandas as pd
from adept import features


###############################################################################
# INPUT
###############################################################################
def read_events(filename, binarize=False, sort=True):
    try:
        df = pd.read_csv(filename, index_col=[0, 1],
                         parse_dates=['event_date'])
    except:
        print "ERROR: Couldn't read {}".format(filename)
        return

    # ignore doc_id. this will not figure into any analyses
    if 'doc_id' in df.columns:
        df.drop('doc_id', axis=1, inplace=True)

    # binarize data, e.g., missing "no asthma" status = False
    if binarize:
        df = (df > 0).astype(int)
       
    if not sort:
        return df
    else:
        return df.sort_index(level=['pt_id', 'event_date'])
    

def read_gold_standard(filename, names=None):
    """
    DEPRECATED import gold standard, return the gold standard and timing
    TODO: should this be a class method?
          depends how i want to use gold_standard and gold_timing
    """
    if not names:
        names=['pt_id', 'event_date', 'event_name']

    df = pd.read_csv(filename, names=names, index_col=[0, 2],
                     parse_dates=['event_date'])
    return df.sort_values('event_date').sort_index(
        level=0, sort_remaining=False)
   

def read_features(filename, names=None, binarize=False,
                  sort=True):
    """
    DEPRECATED read csv input into a pandas dataframe
    """
    if names:
        pass
    else:
        # attempt to determine the corpus name
        for corp_name in features.corpus_feat_map.keys():
            if "features_{}".format(corp_name) in filename:
                names = features.corpus_feat_map[corp_name]
                break
        else:
            # default to pac features, which could fail, of course
            names = features.pac
            print "no appropriate corpus name in filename {}".format(filename)

    col_names = features.meta + names
    
    try:
        df = pd.read_csv(filename, names=col_names, index_col=[0,1],
                         usecols=col_names[:2] + col_names[3:],
                         parse_dates=['event_date'])
    except ValueError:
        print "WARNING: bad csv import. extra comma/columns? column mismatch?"
        print col_names
        df = pd.read_csv(filename, names=col_names, index_col=[0,1],
                         usecols=col_names[:2] + col_names[3:] + ['null'],
                         parse_dates=['event_date'])
    if binarize:
        df = (df > 0).astype(int)
       
    if not sort:
        return df
    else:
        return df.sort_index(level=['pt_id', 'event_date'])

    
def load_corpus(corpus_name="ape35ev", tag="", extrapolator=None,
                binarize=False):
    import os.path

    ft = read_events(
        './srcdata/features_{}.deid.csv'.format(corpus_name),
         binarize=binarize)
    gs = read_events(
        './srcdata/goldtiming_{}.deid.csv'.format(corpus_name))
    if extrapolator:
        gs = extrapolator(ft, gs)

    return ft, gs

    

def events_and_feats(date_diff):
    """
    Introspect on a nested dictionary (e.g., date_diff) object, returning
    the set of possible events and features.
    """
    unique_event_names = set()
    unique_feat_names = set()
    for col_name in date_diff.keys():
        for featval in date_diff[col_name].keys():
            unique_feat_names.add(featval)
            for status in date_diff[col_name][featval].keys():
                unique_event_names.add(status)
    return list(unique_event_names), list(unique_feat_names)


def sample_stats(date_diff, include_no=False):
    """
    Introspect on a nested dictionary (e.g., date_diff) object, returning
    the average number of samples that a pdf would need to estimate 
    """
    counts = []
    states = 0
    for col_name in date_diff.keys():
        for featval in date_diff[col_name].keys():
            for status in date_diff[col_name][featval].keys():
                states += 1
                if include_no or (not include_no and status!='no'):
                    counts.append(len(date_diff[col_name][featval][status]))
    return {'sum': np.sum(counts), 'len': len(counts),
            'mean': np.mean(counts), 'states': states}


###############################################################################
# OUTPUT
###############################################################################
def _arff_header(relation="AsthmaClassification", feature_dict=None,
                 write_ids=True, feature_names=None, valid_values="NUMERIC",
                 valid_classes=['no','asthma']):
    """
    set up arff header, returned as a string
    INPUT: feature_dict is a list of feature names to valid values
    """
    prehead = "@RELATION AsthmaClassification\n\n"
    if write_ids:
        prehead += "@ATTRIBUTE pt_id string\n"
    posthead = "@ATTRIBUTE CLASS {" + ",".join(valid_classes) + "}\n\n@DATA"
    header_content = []

    if feature_dict is not None:
        for name, val in feature_dict.iteritems():
            if isinstance(val, list):
                header_content.append("@ATTRIBUTE {} {}".format(
                    name, "{" + ",".join(str(v) for v in val) + "}"))
            else:
                header_content.append("@ATTRIBUTE {} {}".format(name, val))
    else:
        if valid_values is not None:
            valid_values = "NUMERIC"

        if feature_names is not None:
            # default: from Wu et al, 2014, JAMIA
            feature_names = features.pac

        if isinstance(valid_values, list):
            for name, val in zip(feature_names, valid_values):
                if isinstance(val, list):
                    header_content.append("@ATTRIBUTE {} {}".format(
                        name, "{" + ",".join(str(v) for v in val) + "}"))
                else:
                    header_content.append("@ATTRIBUTE {} {}".format(name, val))
            # header_content = ["@ATTRIBUTE {} {}\n".format(
            #     name, "{" + ",".join(str(v) for v in val) + "}")
            #     for name, val in zip(feature_names, valid_values)]
        else:
            header_content = map(lambda x:
                                 "@ATTRIBUTE {} {}".format(x, valid_values),
                                 feature_names)

    return prehead + "\n" + "\n".join(header_content) + "\n" + posthead


def write_arff(df, output_file=None, relation="AsthmaClassification",
               valid_classes=['no','asthma']):
    # tup = [for df.index.get_values()]
    valid_values = []
    arff_str = ''
    for feat_name, data_type in df.dtypes.iteritems():
        if issubclass(data_type.type, np.integer):
            unique_values = df[feat_name].unique()
            valid_values.append(unique_values)
        elif issubclass(data_type.type, np.float):
            valid_values.append("NUMERIC")
        elif issubclass(data_type.type, np.string_):
            valid_values.append("string")

    arff_str = _arff_header(relation=relation,
                            feature_names=df.columns.values,
                            valid_values=valid_values)
    if not output_file:
        print _arff_header(relation=relation,
                           feature_names=df.columns.values,
                           valid_values=valid_values)

    if isinstance(df.index, pd.core.index.MultiIndex):
        df.index = ["{}_{}".format(pt_id, event_date.strftime("%Y-%m-%d"))
                    for pt_id, event_date in df.index.values]
    if not output_file:
        print df.to_csv(header=False)
    else:
        f = open(output_file, 'wb')
        f.write(arff_str + "\n" + df.to_csv(header=False))


if __name__ == '__main__':

    import glob
    print "Testing whether event-oriented munge loads CSV files correctly"

    for f in glob.glob('srcdata/*ev.*.csv'):
        print " ", f
        read_events(f)
        
    print "Testing whether munge binarizes CSV files without failing"

    for f in glob.glob('srcdata/*ev.*.csv'):
        print " ", f
        read_events(f, binarize=True)
    # gs = read_gold_standard('srcdata/goldtiming_dev127ev.deid.csv')
    # print "Slurped gold standard. Some patients:"
    # print gs.iloc[15:25]

    # df = read_features('srcdata/features_dev127ev.deid.csv')
    # print "\n\nSlurped features.  Some rows:"
    # print df.ix[15:25, ["COUGH", "NIGHTTIME"]]

    # print "\n\nWriting ARFF output"
    # write_arff(df)
    
