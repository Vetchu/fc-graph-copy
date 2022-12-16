import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings


def read_csv(path):
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    import glob
    print("local data", glob.glob("/input/*"))
    print("local data", glob.glob("/mnt/input/*"))

    # read all csv files
    diagnose_df = pd.read_csv("/mnt/input/" + path)  # "/input/DIAGNOSE_ICD_hpo.csv")

    final_df = diagnose_df.copy()
    final_df = final_df[final_df['hpo_features'].apply(lambda x: isinstance(x, str))]
    hpo_string = ';'.join(final_df['hpo_features'])
    hpo_list = hpo_string.split(';')
    hpo_list_unique = list(set(hpo_list))
    final_df[hpo_list_unique] = 0
    final_df.reset_index(inplace=True, drop=True)
    for index, row in final_df.iterrows():
        hpos = row['hpo_features'].split(';')
        for hpo in hpos:
            final_df.loc[index, hpo] = 1
            pass
    final_df.drop(['hpo_features'], axis=1, inplace=True)
    final_df.reset_index(inplace=True, drop=True)

    for index, row in final_df.iterrows():
        final_df.at[index, 'icd9_code'] = np.int64(str(final_df.loc[index, 'icd9_code'])[0])
        if final_df.loc[index, 'icd9_code'] in [9, 0, 5, 7, 3, 1, 6]:
            final_df.at[index, 'icd9_code'] = np.int64(0)

    data_client1, test_data_client1 = train_test_split(final_df, test_size=0.2, random_state=0)
    import os
    os.mkdir('/mnt/output/data')
    data_client1.to_csv('/mnt/output/data/data.csv', index=False)
    test_data_client1.to_csv('/mnt/output/data/test_data.csv', index=False)
    return data_client1

# data_client1, test_data_client1 = train_test_split(client1, test_size=0.2, random_state=0)
# data_client2, test_data_client2 = train_test_split(client2, test_size=0.2, random_state=0)
#
# data_client1.to_csv('fc-random-forest/data/client1/data_client1.csv', index=False)
# data_client2.to_csv('fc-random-forest/data/client2/data_client2.csv', index=False)
# test_data_client1.to_csv('fc-random-forest/data/client1/test_data_client1.csv', index=False)
# test_data_client2.to_csv('fc-random-forest/data/client1/test_data_client1.csv', index=False)
