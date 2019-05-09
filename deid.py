import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random, string
from collections import defaultdict

dir = "srcdata/"
id_offsets = [11111111, 22222222, 33333333]
dt_offsets = [111, 222, 333]

def deid(dataset, id_offset=1111111111111, date_offset=111):
    out_feats = open(dir + "features_" + dataset + ".deid.csv", 'wb')
    ids = defaultdict()
    ids.default_factory = lambda : len(ids) + id_offset
    
    with open(dir + "features_" + dataset + ".csv") as feats_f:
        for line in feats_f:
            parts = line.split(",")
            if not parts:
                continue

            new_parts = []
            new_parts.append(str(ids[int(parts[0])]))
            new_parts.append((pd.to_datetime(parts[1])
                             + timedelta(days=date_offset))
                            .strftime("%Y-%m-%d"))
            new_parts.append(''.join(random.choice(string.digits)
                                     for _ in range(8)))
            new_parts.extend(parts[3:])
            out_feats.write(",".join(new_parts))

    out_gold = open(dir + "goldtiming_" + dataset + ".deid.csv", 'wb')
    with open(dir + "goldtiming_" + dataset + ".csv") as gold_f:
        for line in gold_f:
            parts = line.split(",")
            if not parts:
                continue
            new_parts = []
            new_parts.append(str(ids[int(parts[0])]))
            try:
                new_parts.append((pd.to_datetime(parts[1])
                                  + timedelta(days=date_offset))
                                 .strftime("%Y-%m-%d"))
            except ValueError:
                new_parts.append('')
            new_parts.extend(parts[2:])
            print ",".join(new_parts)
            out_gold.write(",".join(new_parts))

if __name__=="__main__":
    from sys import argv
    if len(argv) < 2:
        datasets = ["dev127", "test112", "ex001"]
    else:
        datasets = argv[1].split(',')

    for dataset, id_offset in zip(datasets, id_offsets):
        deid(dataset, id_offset)
