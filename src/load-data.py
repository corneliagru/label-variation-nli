import os

print(os.getcwd())
import pandas as pd

import config

from custom_funcs import load_data_as_df


# load jsonl as df
df_snli = load_data_as_df(data_file=config.CHAOSNLI_SNLI)

# combine original df, with extracted premise and hypothesis
df_snli = pd.concat([df_snli.drop(['example'], axis=1), df_snli['example'].apply(pd.Series)], axis=1)

#check whether both 'uid' columns are the same, then drop the second
if df_snli.iloc[:, 0].equals(df_snli.iloc[:, 8]):
    df_snli = df_snli.loc[:, ~df_snli.columns.duplicated()].copy()

#drop sources column, as source is always "source snli_agree_3"
if df_snli['source'].nunique() == 1:
    df_snli.drop(['source'], axis=1, inplace=True)

# extract one hot labels
df_snli = pd.concat([df_snli.drop(['label_counter'], axis=1), df_snli['label_counter'].apply(pd.Series)], axis=1)

#replace nan with 0 counts
df_snli['n'] = df_snli['n'].fillna(0)
df_snli['e'] = df_snli['e'].fillna(0)
df_snli['c'] = df_snli['c'].fillna(0)

#personal ground truth of annotator that came up with hypothesis
df_snli['ground_truth'] = df_snli['old_labels'].str[0].str[0]


df_snli.to_csv('../data/final/snli-clean.csv')

print("data saved to ../data/final/snli-clean.csv")