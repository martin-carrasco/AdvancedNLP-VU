import json
import pandas as pd


def preprocess():
    with open('test_data/raw_test_data.json') as f:
        data = json.load(f)
        del data['example_idx']

        df = pd.DataFrame.from_dict(data)

        df['capability'] = df['test_name'].str.split(' - ').str[0]

        df.to_csv('test_data/challenge_ds.csv', index=False)



if __name__ == "__main__":
    preprocess()