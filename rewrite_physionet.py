# -*- coding: utf-8 -*-
"""
    File name: rewrite_physionet.py
    Project: 
    Desciption: 
    Author: Sijia Liu (m142167)
    Date created: Sep 20, 2017
"""

import os
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
import datetime

ad_date = datetime.date(2000, 1, 1)


def get_dict_vectorizer(in_root):
    total_rows = []

    # first pass: generate dictionary vectorizer
    for doc in os.listdir(in_root):
        record_dict = defaultdict(dict)
        skip_head = True

        with open(os.path.join(in_root, doc)) as f:
            for line in f:
                """
                Time,Parameter,Value
                00:00,RecordID,132539
                00:00,Age,54
                00:00,Gender,0
                00:00,Height,-1
                00:00,ICUType,4
                00:00,Weight,-1
                """
                if skip_head:
                    skip_head = False
                    continue

                (time, parameter, value) = line.strip().split(',')
                # 01:37 -> 1 * 60 + 37
                minutes = int(time[0:2]) * 60 + int(time[3:5])
                if parameter == 'RecordID':
                    continue
                record_dict[parameter] = float(value)

        total_rows.append(record_dict)

    dict_vectorizer = DictVectorizer(sparse=False)
    dict_vectorizer.fit(total_rows)

    print "Total features: \t%d" % len(dict_vectorizer.feature_names_)
    # print v.inverse_transform(X)
    # print dict_vectorizer.feature_names_

    return dict_vectorizer


def rewrite_phyisonet_adept_features(in_root, out_path, dict_vectorizer):
    fo = open(out_path, 'w')

    # write headline
    fo.write("pt_id,event_date,doc_id," + ','.join(dict_vectorizer.feature_names_) + '\n')

    doc_id = 10e6       # fake doc id. Acutally time stamp ID
    # second pass: use dict_vectorizer to generate features
    for doc in os.listdir(in_root):
        record_dict = defaultdict(dict)
        skip_head = True

        with open(os.path.join(in_root, doc)) as f:

            patient_id = doc[:-4]

            for line in f:
                if skip_head:
                    skip_head = False
                    continue

                (time, parameter, value) = line.strip().split(',')
                # 01:37 -> 1 * 60 + 37
                minutes = int(time[0:2]) * 60 + int(time[3:5])

                if parameter == 'RecordID':
                    continue
                record_dict[minutes][parameter] = float(value)

        # done reading, write out patient features with fake document id
        for minutes, row_dict in record_dict.iteritems():
            doc_id += 1
            features = dict_vectorizer.transform(row_dict)[0]

            # TODO: check fake date options. Currently I used number of hours to replace dates.
            fake_date = ad_date + datetime.timedelta(days=int(minutes / (60)))
            fo.write("%s,%s,%d,%s\n" % (patient_id, fake_date, doc_id,
                                        ','.join(['%.2f' % v for v in features])))

    fo.close()


def rewrite_physionet_gs(gs_txt, output_txt):
    skip_head = True

    fo = open(output_txt, 'w')
    fo.write('pt_id,event_date,asthma_status\n')

    with open(gs_txt) as f:
        for line in f:
            """
            Time,Parameter,Value
            00:00,RecordID,132539
            00:00,Age,54
            00:00,Gender,0
            00:00,Height,-1
            00:00,ICUType,4
            00:00,Weight,-1
            """
            if skip_head:
                skip_head = False
                continue

            (pid, saps, sofa, length_of_stay, survival, death) = line.strip().split(',')

            fo.write("%s,%s,%s\n" %(pid, ad_date, 'hospitalization'))

            if int(death) == 0:
                fo.write("%s,%s,%s\n" % (pid, ad_date + datetime.timedelta(days=int(length_of_stay)), 'discharge'))
            else:
                fo.write("%s,%s,%s\n" % (pid, ad_date + datetime.timedelta(days=int(survival)), 'death'))

    fo.close()


def rewrite_physionet_gs(gs_txt, output_txt):
    skip_head = True

    fo = open(output_txt, 'w')
    fo.write('pt_id,event_date,asthma_status\n')

    with open(gs_txt) as f:

        """
        RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death
        132539,6,1,5,-1,0
        132540,16,8,8,-1,0
        132541,21,11,19,-1,0
        132543,7,1,9,575,0
        """

        for line in f:
            if skip_head:
                skip_head = False
                continue

            (pid, saps, sofa, length_of_stay, survival, death) = line.strip().split(',')

            fo.write("%s,%s,%s\n" %(pid, ad_date, 'hospitalization'))

            if int(death) == 0:
                fo.write("%s,%s,%s\n" % (pid, ad_date + datetime.timedelta(days=int(length_of_stay)), 'discharge'))
            else:
                fo.write("%s,%s,%s\n" % (pid, ad_date + datetime.timedelta(days=int(survival)), 'death'))

    fo.close()


def rewrite_physionet_length_stay_larger_3(gs_txt, output_txt):
    skip_head = True

    fo = open(output_txt, 'w')
    fo.write('pt_id,event_date,asthma_status\n')

    with open(gs_txt) as f:

        """
        RecordID,SAPS-I,SOFA,Length_of_stay,Survival,In-hospital_death
        132539,6,1,5,-1,0
        132540,16,8,8,-1,0
        132541,21,11,19,-1,0
        132543,7,1,9,575,0
        """

        for line in f:
            if skip_head:
                skip_head = False
                continue

            (pid, saps, sofa, length_of_stay, survival, death) = line.strip().split(',')

            fo.write("%s,%s,%s\n" % (pid, ad_date, 'hospitalization'))

            if int(death) == 0:
                fo.write("%s,%s,%s\n" % (pid, ad_date + datetime.timedelta(days=int(length_of_stay)), 'discharge'))
            else:
                fo.write("%s,%s,%s\n" % (pid, ad_date + datetime.timedelta(days=int(survival)), 'death'))

    fo.close()


if __name__ == '__main__':
    in_root = '../data/set-a'
    dict_vectorizer = get_dict_vectorizer(in_root)
    rewrite_phyisonet_adept_features(in_root, "../adept_data/features_physionet2012.deid.csv", dict_vectorizer)

    # rewrite_physionet_gs('../data/Outcomes-a.txt', '../adept_data/goldtiming_physionet2012.deid.csv')