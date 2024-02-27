import pandas as pd
from typing import Tuple
import fasttext.util

def embedding_or_empty(word, model):
    """  Get the word embedding from the model or return a zero vector if the word is not in the model
    """
    try:
        return model[word]
    except:
        return [0]*300

def feature_encode(df: pd.DataFrame, task_ident: bool) -> Tuple:
    """ Encode the features and return the X and y
    """
    fasttext.util.download_model('en', if_exists='ignore')
    model = fasttext.load_model('data/models/cc.en.300.bin')

    # Add one-hot encoded columns for pos
    one_hot_df = pd.get_dummies(df['POS'], prefix='POS')
    df = pd.concat([df, one_hot_df], axis=1)

    # Add one-hot encoded columns for deprel
    one_hot_df = pd.get_dummies(df['deprel'], prefix='deprel')
    df = pd.concat([df, one_hot_df], axis=1)

    # Add one-hot encoded columns for phrase_type
    one_hot_df = pd.get_dummies(df['phrase_type'], prefix='phrase_type')
    df = pd.concat([df, one_hot_df], axis=1)

    df.drop(['POS', 'deprel', 'phrase_type'], axis=1, inplace=True)
    #  Create list of feature columns
    feat_col = [col for col in df if col.startswith('phrase_type') or col.startswith('POS') or col.startswith('deprel')]

    feat_col = feat_col + ['position_rel_left', 'position_rel_right', 'voice_active', 'voice_passive']

    df['token'] = df['token'].apply(lambda x: embedding_or_empty(x, model))

    embedding_cols = [f'embedding_{i}' for i in range(len(df['token'][0]))]

    df1 = pd.DataFrame(df['token'].to_list(), columns=embedding_cols)
    df.drop(['token'], axis=1, inplace=True)
    df = pd.concat([df, df1], axis=1)

    df['label_identification'] = df['label'].apply(lambda x: 1 if x != 'O' and x != 'V' else 0)

    X = df[feat_col + embedding_cols]
    if task_ident:
        y = list(df['label_identification'])
    else:
        y = list(df['label'])

    return X, y
