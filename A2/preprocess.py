from sklearn.linear_model import LogisticRegression
from conllu import parse
import pandas as pd



conll_dir = 'data/'
test_file = conll_dir + 'en_ewt-up-test.conllu' 
train_file = conll_dir + 'en_ewt-up-train.conllu' 
dev_file = conll_dir + 'en_ewt-up-dev.conllu' 



df_dict = {
    'token_id': [],
    'sentence_num': [],
    'word': [],
    'lemma': [],
    'upos': [],
    'xpos': [],
    'feats': [],
    'head': [],
    'deprel': [],
    'deps': [],
    'misc': [],
    'up:preds': [],
    'up:args': []
}

with open(dev_file, 'r') as f:
    sent_cnt = -1
    for line in f.readlines():
        line = line.strip()
        if line.startswith('#'):
            continue
        columns = line.split('\t')
        if len(columns) < 2:
            continue

        token_id = columns[0]
        if token_id == '1':
            sent_cnt+=1
        df_dict['sentence_num'].append(sent_cnt)
        df_dict['token_id'].append(columns[0])
        df_dict['word'].append(columns[1])
        df_dict['lemma'].append(columns[2])
        df_dict['upos'].append(columns[3])
        df_dict['xpos'].append(columns[4])
        df_dict['feats'].append(columns[5])
        df_dict['head'].append(columns[6])
        df_dict['deprel'].append(columns[7])
        df_dict['deps'].append(columns[8])
        df_dict['misc'].append(columns[9])
        if len(columns) > 10:
            df_dict['up:preds'].append(columns[10])
            df_dict['up:args'].append(columns[11:])
        else:
            df_dict['up:preds'].append('_')
            df_dict['up:args'].append(['_'])
df = pd.DataFrame(df_dict)


def preprocessing(df):
    token_df_dict = {
        'token': [],
        'lemma': [],
        'pos': [],
        'predicate': [],
        'srl': [],
        'deprel': []
    }
    sentence_ids = df['sentence_num'].unique()
    # For each senttence
    for sentence_id in sentence_ids:
        sentence_df = df[df['sentence_num'] == sentence_id]
        # Extract the predicates of this sentence
        predicates = list(sentence_df['up:preds'].unique())
        # Remove empty predicate
        if '_' in predicates:
            predicates.remove('_')
        
        for pred in predicates:
            # Iterate over the tokens which are not predicates
            for i, row in sentence_df[sentence_df['up:preds'] == '_'].iterrows():
                if row['upos'] == 'PUNCT':
                    continue
                token_df_dict['token'].append(row['word'])
                token_df_dict['lemma'].append(row['lemma'])
                token_df_dict['pos'].append(row['upos'])
                token_df_dict['deprel'].append(row['deprel'])


                if all(p=='_'for p in row['up:args']):
                    token_df_dict['predicate'].append(False)
                    token_df_dict['srl'].append(False)
                else:
                    token_df_dict['predicate'].append(pred)
                    token_df_dict['srl'].append(row['up:args'][predicates.index(pred)])
    return pd.DataFrame(token_df_dict)

n_df = preprocessing(df)

n_df.to_csv('data/dev.csv', index=False)