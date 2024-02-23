from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd

from utils import feature_encode


def train_model(train_file: str):
    """  Train the model and save it to a file
    """
    df = pd.read_csv(train_file, delimiter='\t')
    X, y = feature_encode(df)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
    pickle.dumps(clf, open('data/models/model_ident.pkl', 'wb'))
    print('Model saved to model_ident.pkl')


def predict_model(test_file: str, model_file: str):
    X, y = feature_encode(test_file)
    clf = pickle.loads(open(f'data/models/{model_file}', 'rb'))
    y_pred = clf.predict(X)
    df=pd.DataFrame({'y': y, 'pred_y': y_pred})
    df.to_csv('data/preds/ident.csv', index=False)
    print('Predictions saved to ident.csv')



