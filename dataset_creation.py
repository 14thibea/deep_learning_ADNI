import pandas as pd
from os import path

train_prop = 0.9

diagnosis_labels = ['CN', 'AD']
cohorts = ['ADNI', 'AIBL', 'OASIS']
dataset_name = 'complete4'
data_path = path.join('/Volumes/aramis-projects/elina.thibeausutre/data', dataset_name)
filename = dataset_name + '_diagnoses.tsv'
data_df = pd.read_csv(path.join(data_path, filename), sep='\t')

train = []
test = []

for cohort in cohorts:
    for diagnosis in diagnosis_labels:
        diagnosis_df = data_df[(data_df.diagnosis == diagnosis) &
                               (data_df.cohort == cohort)]
        n_diagnosis = len(diagnosis_df)
        train.append(diagnosis_df.iloc[:int(train_prop * n_diagnosis):])
        test.append(diagnosis_df.iloc[int(train_prop * n_diagnosis)::])

train_df = pd.concat(train)
test_df = pd.concat(test)

train_filename = dataset_name + '_train.tsv'
test_filename = dataset_name + '_test.tsv'
train_df.to_csv(path.join(data_path, train_filename), sep='\t', index=False)
test_df.to_csv(path.join(data_path, test_filename), sep='\t', index=False)
