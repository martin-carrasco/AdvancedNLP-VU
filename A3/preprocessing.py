from transformers import AutoTokenizer
import pandas as pd
from utils import check_label


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

raw_dir = 'data/raw/'
df_dict = {
    'token_id': [],
    'sentence_num': [],
    'token': [],
    'lemma': [],
    'upos': [],
    'POS': [],
    'feats': [],
    'head': [],
    'deprel': [],
    'deps': [],
    'misc': [],
    'up:preds': [],
    'up:args': []
}

def parse(filename: str) -> pd.DataFrame:
    """ Parse a conllu file and return a dataframe with the parsed data
    """
    with open(raw_dir + filename, 'r') as f:
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
            df_dict['token'].append(columns[1])
            df_dict['lemma'].append(columns[2])
            df_dict['upos'].append(columns[3])
            df_dict['POS'].append(columns[4])
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
    return pd.DataFrame(df_dict)

def preprocessing(filename: str) -> pd.DataFrame:
    """  Preprocess the conllu file and return a dataframe with the preprocessed data
    """
    df = parse(filename)

    df_list = []

    # Get unique sentence ids
    sentence_ids = df['sentence_num'].unique()

    # Initialize columns
    df['sentence_num_clone'] = 0
    df['predicate'] = df['up:preds']
    new_sentence_id = df['sentence_num'].max() + 1

    # For each senttence
    for sentence_id in sentence_ids:
        sentence_df = df[df['sentence_num'] == sentence_id]

        # Extract the predicates of this sentence
        predicates = []
        predicate_tokens = []
        predicate_token_ids = []

        # Extract the actual tokens of the predicates
        for i, row in sentence_df.iterrows():
            if row['up:preds'] == '_':
                continue
            predicate_tokens.append(row['token'])
            predicates.append(row['up:preds'])
            predicate_token_ids.append(row['token_id'])


         # _ is not a prediate
        if '_' in predicates:
            predicates.remove('_')

        # Create a new sentence based on predicate
        for i, tup in enumerate(zip(predicates, predicate_tokens, predicate_token_ids)):
            pred, pred_token, pred_token_id = tup
            new_sentence_df = sentence_df.copy()
            new_sentence_df['sentence_num_clone'] = new_sentence_id if i != 0 else sentence_id
            new_sentence_df['predicate'] = pred
            new_sentence_df['predicate_token'] = pred_token
            new_sentence_df['predicate_token_id'] = pred_token_id
            new_sentence_df['label'] = new_sentence_df['up:args'].apply(lambda x: check_label(x, i))

            # Add DF subsection
            df_list.append(new_sentence_df)

            if i != 0:
                new_sentence_id += 1
    new_df = pd.concat(df_list)
    new_df['sentence_id'] = new_df['sentence_num_clone']
    new_df = new_df.drop(['sentence_num_clone'], axis=1)

    file_name = filename.split('.')[0]
    new_df.to_csv(f'data/preprocessed/{file_name}_pp.tsv', index=False, sep='\t')
    return new_df


#df = preprocessing('en_ewt-up-train.conllu')
#new_df =  df.groupby('sentence_num')[['token']].apply(list)
#print(new_df.columns)
#print(new_df.head(5)['predicate'])


# predicate = 'AA'
# tokenized_input = tokenizer([text_list, predicate], is_split_into_words=True)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])