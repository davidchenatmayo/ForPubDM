# original PAC algorithm features from  Wu et al 2014 JAMIA article
pac = ["ASTHMA", "COPD", "BRONCHOSPASM", "COUGH", "WHEEZE", "DYSPNEA",
       "CRITERIA1", "NIGHTTIME", "NONSMOKER", "NASAL_POLYPS",
       "EOSINOPHILIA_HIGH", "POSITIVE_SKIN", "SERUM_IGE_HIGH",
       "HAY_FEVER", "INFANTILE_ECZEMA", "EXPOSURE_TO_ANTIGEN",
       "PULMONARY_TEST", "FEV1_INCREASE", "FVC_INCREASE",
       "FEV1_DECREASE", "PULMONARY_LOW_IMPROVEMENT",
       "METHACHOLINE_TEST", "METHACHOLINE_FEV1_LOW", "BRONCHODILATOR",
       "FAVORABLE_RESPONSE", "BRONCHODILATOR_RESPONSE"]

# APE algorithm features with remission and relapse
ape = ["ASTHMA_MEDICATION", "ASTHMA_VISIT", "ASTHMA_SYMPTOM",
       "NO_SMOKING_EXPOSURE", "POSITIVE_SKIN", "NEGATIVE_SKIN",
       "HAY_FEVER", "INFANTILE_ECZEMA", "EXPOSURE_TO_ANTIGEN",
       "PULMONARY_TEST_NORMAL", "FVC_NORMAL", "FVC1_NORMAL",
       "FVC1_INCREASED", "BRONCHODILATOR_RESPONSE",
       "METHACHOLINE_FEV1_LOW", "FAMILY_HISTORY_OF_ATOPIC_CON",
       "ASTHMA_SEVERITY", "DOSE_ADJUSTMENT", "ASTHMA_CONTROL",
       "FENO_INCREASED", "OVERWEIGHT", "POOR_COMPLIANCE",
       "TYPE_ASTHMA", "VIRAL_INFECTION",
       # negative/conditional/hypothetical/etc
                "NOT_ASTHMA_MEDICATION", "NOT_ASTHMA_VISIT",
       "NOT_ASTHMA_SYMPTOM", "NOT_NO_SMOKING_EXPOSURE",
       "NOT_POSITIVE_SKIN", "NOT_NEGATIVE_SKIN", "NOT_HAY_FEVER",
       "NOT_INFANTILE_ECZEMA", "NOT_EXPOSURE_TO_ANTIGEN",
       "NOT_PULMONARY_TEST_NORMAL", "NOT_FVC_NORMAL",
       "NOT_FVC1_NORMAL", "NOT_FVC1_INCREASED",
       "NOT_BRONCHODILATOR_RESPONSE", "NOT_METHACHOLINE_FEV1_LOW",
       "NOT_FAMILY_HISTORY_OF_ATOPIC_CON", "NOT_ASTHMA_SEVERITY",
       "NOT_DOSE_ADJUSTMENT", "NOT_ASTHMA_CONTROL",
       "NOT_FENO_INCREASED", "NOT_OVERWEIGHT", "NOT_POOR_COMPLIANCE",
       "NOT_TYPE_ASTHMA", "NOT_VIRAL_INFECTION"]

corpus_feat_map = {'dev127': pac,
                   'test112': pac,
                   'ape35': ape}

meta = ['pt_id', 'event_date', 'doc_id']
exclude = ['doc_id']
