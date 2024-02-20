from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import gensim.models.keyedvectors as word2vec
import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')


model = fasttext.load_model('cc.en.300.bin')
# model = word2vec.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)








conll_dir = 'data/'
test_file = conll_dir + 'en_ewt-up-test.conllu' 
train_file = conll_dir + 'en_ewt-up-train.conllu' 
dev_file = conll_dir + 'en_ewt-up-dev.conllu' 


df = pd.read_csv('data/dev.csv')
df['srl_iden'] = df['predicate'].apply(lambda x: True if x != 'False' else False)

le = LabelEncoder()
df['pos'] = le.fit_transform(df['pos'])
df['deprel'] = le.fit_transform(df['deprel'])
df[['lemma', 'token']] = df[['lemma', 'token']].astype(str)



df['lemma'] = df['lemma'].apply(lambda x: model[x])
df['token'] = df['token'].apply(lambda x: model[x])


df['X'] = df['lemma'] + df['token'] + df['pos'] + df['deprel']

X = list(df['X'])
y = list(df['srl_iden'])


clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y)
score = cross_val_score(clf, X, y, cv=5, scoring='f1')
print(score)


