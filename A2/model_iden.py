from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import pandas as pd

from utils import feature_encode


def train_model(train_file: str):
    """  Train the model and save it to a file
    """
    df = pd.read_csv('data/input/' + train_file, delimiter='\t')
    X, y = feature_encode(df, task_ident=True)
    print('Training model ...')
    clf = LogisticRegression(random_state=0).fit(X, y)
    print('Saving model ...')
    dump(clf, 'data/models/model_ident.joblib')
    print('Model saved to model_ident.joblib')


def predict_model(test_file: str, model_file: str):
    df = pd.read_csv('data/input/' + test_file, delimiter='\t')
    X, y = feature_encode(df, task_ident=True)
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
    df.to_csv('data/preds/ident.csv', index=False)
    print('Predictions saved to ident.csv')



