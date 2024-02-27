from sklearn.linear_model import LogisticRegression
from joblib import dump, load

import pandas as pd

from utils import feature_encode

def train_model(train_file: str):
    """  Train the model and save it to a file
    """
    df = pd.read_csv('data/input/' + train_file, delimiter='\t')
    df = df[(df['label'] != 'O') | (df['label'] != 'V')]
    X, y = feature_encode(df, task_ident=False)
    clf = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial').fit(X, y)
    dump(clf, 'data/models/model_classf.joblib')
    print('Model saved to model_classf.joblib')

def predict_model(test_file: str, model_file: str):
    """  Predict the model and save it to a file
    """
    df = pd.read_csv('data/input/' + test_file, delimiter='\t')
    df = df[(df['label'] != 'O') | (df['label'] != 'V')]
    X, y = feature_encode(df, task_ident=False)
    clf = load(f'data/models/{model_file}')

    features_in_data = X.columns
    allowed_features = clf.feature_names_in_

    common_features = set(features_in_data).intersection(allowed_features)

    if len(list(common_features)) < len(allowed_features):
        missing_features = list(set(allowed_features) - set(common_features))
        print('Some features are missing in the data')
        print('Missing features: ', missing_features)
        X[missing_features] = 0

    X = X[allowed_features]

    y_pred = clf.predict(X)
    df=pd.DataFrame({'y': y, 'pred_y': y_pred})
    df.to_csv('data/preds/classf.csv', index=False)
    print('Predictions saved to classf.csv')