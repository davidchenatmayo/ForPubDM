from adept import estimate as est

def sum_aggregate(features, statuses_per_patient='single'):
    if statuses_per_patient.startswith(('M', 'm')): # multiple
        return features.cumsum(axis=0)
    else:
        return features.sum(level=0)


def or_aggregate(features, statuses_per_patient='single'):
    if statuses_per_patient.startswith(('M', 'm')):
        return features.cummax(axis=0)
    else:
        return features.any(level=0).astype(int)
    

def pdfin_aggregate(features, event='asthma',
                    train_corpus='dev127ev', tag='', force=False,
                    binarize=False):
    """
    pdfin_aggregate - Use probability density functions (PDFs) with:
     - features vs gold standard on the train_corpus
     - no-op extrapolation
     - regular diff
    """
    pdfs = est.get_pdfs_for_corpus(train_corpus, tag=tag, force=force,
                                   binarize=binarize)
    agg_ft = features.copy()

    ## TODO: what should this do in a non-binary feature case?
    eventA = 1
    ## TODO: what should this do in a disease progression case?
    eventB = event

    for f_name in features.columns:
        # Exclude if insufficient support for this feature
        if f_name not in pdfs.keys():
            agg_ft.drop(f_name, axis=1, inplace=True)
            continue
        for pt_id in features.index.levels[0]:                
            instances = agg_ft.loc[pt_id, f_name].copy()
            agg_inst = []
            events = []
            ## TODO: what if those events don't show up?
            if (eventA, eventB) in pdfs[instances.name].keys():
                pdf = pdfs[instances.name][(eventA, eventB)]
            else:
                agg_ft.drop(f_name, axis=1, inplace=True)
                break
            for date_a, event_a in instances.iteritems():
                if event_a:
                    events.append((date_a, event_a))
                contrib = np.array([pdf[(date_b - date_a).days]
                                    for date_b, event_b in events])
                agg_inst.append(np.sum(contrib))
            agg_ft.loc[pt_id, f_name] = agg_inst
    return agg_ft


def pdfst_aggregate(features, event='asthma',
                    train_corpus='dev127ev', tag='', force=False,
                    binarize=False):
    """
    pdfst_aggregate - Use probability density functions (PDFs) with:
     - features vs gold standard on the train_corpus
     - timing extrapolation of event beginnings (extrap_begin)
     - regular diff
    """
    from adept import extrap
    pdfs = est.get_pdfs_for_corpus(train_corpus, tag=tag, force=force,
                                   binarize=binarize,
                                   extrapolator=extrap.extrap_begin)
    agg_ft = features.copy()

    ## TODO: what should this do in a non-binary feature case?
    eventA = 1
    ## TODO: what should this do in a disease progression case?
    eventB = event

    for f_name in features.columns:
        # Exclude if insufficient support for this feature
        if f_name not in pdfs.keys():
            agg_ft.drop(f_name, axis=1, inplace=True)
            continue
        for pt_id in features.index.levels[0]:                
            instances = agg_ft.loc[pt_id, f_name].copy()
            agg_inst = []
            events = []
            ## TODO: what if those events don't show up?
            if (eventA, eventB) in pdfs[instances.name].keys():
                pdf = pdfs[instances.name][(eventA, eventB)]
            else:
                agg_ft.drop(f_name, axis=1, inplace=True)
                break
            for date_a, event_a in instances.iteritems():
                if event_a:
                    events.append((date_a, event_a))
                contrib = np.array([pdf[(date_b - date_a).days]
                                    for date_b, event_b in events])
                agg_inst.append(np.sum(contrib))
            agg_ft.loc[pt_id, f_name] = agg_inst
    return agg_ft


if __name__ == '__main__':
    import timeit
    from adept import munge as mng
    from adept import extrap
    ft = mng.read_events('srcdata/features_dev127ev.deid.csv', binarize=True)
    gs = mng.read_events('srcdata/goldtiming_dev127ev.deid.csv')

    sum_features = sum_aggregate(ft)
    or_features = or_aggregate(ft)

    t0 = timeit.default_timer()
    pdfin_features = pdfin_aggregate(ft, force=False, train_corpus='dev127ev',
                                     tag='indexdate-binaryfeat', binarize=True)
    print "{} elapsed, pdf-in agg".format(timeit.default_timer() - t0)
    # pdfst_features = pdfst_aggregate(ft, force=False, train_corpus='dev127ev',
    #                                  tag='eventdate-binaryfeat', binarize=True)
    # print "{} elapsed, pdf-st agg".format(timeit.default_timer() - t0)

    # TODO: make the "extrap" frameowork fit this! and rewrite merge_gold
    mng.write_arff(extrap.merge_gold(sum_features, gs),
                   output_file='gendata/dev127.pysum.ssp.arff',
                   relation="SumAsthmaClassification")
    mng.write_arff(extrap.merge_gold(or_features, gs),
                   output_file='gendata/dev127.pyor.ssp.arff',
                   relation="OrAsthmaClassification")
    # TODO: does merge_gold work here still?
    # TODO: is this legitimately SSP?
    mng.write_arff(extrap.merge_gold(pdfin_features, gs),
                   output_file='gendata/dev127.pypdfin.ssp.arff',
                   relation="PdfInAsthmaClassification")
    # mng.write_arff(extrap.merge_gold(pdfst_features, gs),
    #                output_file='gendata/dev127.pypdfst.ssp.arff',
    #                relation="PdfStAsthmaClassification")
