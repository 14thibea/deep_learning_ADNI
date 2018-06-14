import pandas as pd
import numpy as np
from os import path, sep


parser = argparse.ArgumentParser()
parser.add_argument("tsv_path", type=str,
                    help='path to your list of subjects in a tsv file')
parser.add_argument("--t", "train_prop", type=float, default=0.9,
                    help="proportion of subjects in the training set")
args = parser.parse_args()

train_prop = args.train_prop
data_path = sep.join(args.tsv_path.split(sep)[:-1:])
dataset_name = args.tsv_path.split(sep)[-1].split('.')[0]
data_df = pd.read_csv(args.tsv_path, sep='\t')

cohorts = np.unique(data_df["cohort"]).tolist()
diagnosis_labels = np.unique(data_df['diagnosis']).tolist()

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
