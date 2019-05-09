import numpy as np
import pandas as pd

### which values in a tensor should be treated differently?
MASK_VALUE = -1
STOP_VALUE = 99

### what are the other values to expect in a tensor, and what do they mean?
default_classes = ['no', 'asthma', 'remission', 'relapse']
default_class_index = {'no': np.array([1,0,0,0]),
                       'asthma': np.array([0,1,0,0]),
                       'remission': np.array([0,0,1,0]),
                       'relapse': np.array([0,0,0,1])}

physionet_classes = ['no', 'hospitalization', 'discharge', 'death']
physionet_class_index = {'no': np.array([1,0,0,0]),
                         'hospitalization': np.array([0,1,0,0]),
                         'discharge': np.array([0,0,1,0]),
                         'death': np.array([0,0,0,1])}
