from sklearn.linear_model import LogisticRegression
from joblib import dump, load
import pandas as pd

from utils import feature_encode


def train_model(train_file: str):
    """  Train the model and save it to a file
    """
    df = pd.read_csv('data/input/' + train_file, delimiter='\t')
    X, y = feature_encode(df)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
    dump(clf, 'data/models/model_ident.joblib')
    print('Model saved to model_ident.joblib')


def predict_model(test_file: str, model_file: str):
    df = pd.read_csv('data/input/' + test_file, delimiter='\t')
    X, y = feature_encode(df)
    clf = load(f'data/models/{model_file}')
    y_pred = clf.predict(X)
    df=pd.DataFrame({'y': y, 'pred_y': y_pred})
    df.to_csv('data/preds/ident.csv', index=False)
    print('Predictions saved to ident.csv')



