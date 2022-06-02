import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def preproc_narti(df):
    df = pd.read_csv('benchmarking/result/result_NARTI_tree.csv',
                            index_col=0).drop(['type'], axis=1)
    df = df[df.method == 'modified_map']
    df.method = 'NARTI'
    df.loc[df['source'] == 'dyno', 'source'] = 'real'
    return df


fps = [
    'benchmarking/result/result_other_methods.csv',
    'benchmarking/result/result_narti.csv'
]

df = pd.concat([pd.read_csv(fp) for fp in fps])

sns.barplot(data=df)