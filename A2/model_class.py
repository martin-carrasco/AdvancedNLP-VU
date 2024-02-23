from sklearn.linear_model import LogisticRegression

import pandas as pd
import pickle

from utils import feature_encode

def train_model(train_file: str):
    """  Train the model and save it to a file
    """
    df = pd.read_csv(train_file, delimiter='\t')
    df = df[(df['label'] != 'O') | (df['label'] != 'V')]
    X, y = feature_encode(df)
    clf = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial').fit(X, y)
    pickle.dumps(clf, open('data/models/model_classf.pkl', 'wb'))

def predict_model(test_file: str, model_file: str):
    """  Predict the model and save it to a file
    """
    df = pd.read_csv(test_file)
    df = df[(df['label'] != 'O') | (df['label'] != 'V')]
    X, y = feature_encode(df)

    clf = pickle.load(open(f'data/models/{model_file}', 'rb'))
    y_pred = clf.predict(X)
    df=pd.DataFrame({'y': y, 'pred_y': y_pred})
    df.to_csv('data/preds/classf.csv', index=False)
    print('Predictions saved to classf.csv')