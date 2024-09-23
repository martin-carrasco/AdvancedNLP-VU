from transformers import AutoTokenizer
import os
from typing import List, Dict
import pandas as pd


raw_dir = 'data/raw/'

def remove_prefix(text: str):
    """ Removes the prefixes from the labels.


        Parameters
        ----------
        text : str
            The label to remove the prefix from.

        Returns
        -------
        str
            The label without the prefix.    

    
    """
    text_cp = text
    if 'C-' in text:
        text_cp = text_cp.replace('C-', '')
    if 'R-' in text:
        text_cp = text_cp.replace('R-', '')
    # Replace the _ with O
    return text_cp.replace('_', 'O')

def check_label(x: str, i: int) -> str:
    """ Check the label of the token and return the label for the predicate i.


        Parameters
        ----------
        x : str
            The label of the token.
        i : int
            The index of the predicate.
        
        Returns
        -------
        str
            The label of the token for the predicate i.
    
    """
    try:
        label = x[i]
        return label if label != '_' else 'O'
    except:
        return 'O'

class InvalidTokenException(Exception):
    pass

class ConLLToken:
    def __init__(self, data: List, element_idx, e_id):
        self.data = data # Save original data

        try:
            self.id = int(data[0])
        except: 
            raise InvalidTokenException(f'Invalid token id: {data[0]}')
        
        self.position = element_idx # Index in sentence
        self.token = data[1]
        self.lemma = data[2]
        self.pos_u = data[3]
        self.pos_tag = data[4]
        self.d_tag = data[5]
        self.head = data[6]
        self.dep_tag = data[7]
        self.is_pred = False
        self.pred = None
        self.labels = []

        # Predicate checking
        if len(data) > 10 and data[10] != '_':
            self.is_pred = True
            self.pred = data[10]

        # Argument checking
        if len(data) > 11:
            self.labels = [remove_prefix(p) for p in data[11:]]

    def __str__(self):
        return self.token
    def __repr__(self):
        return self.token

class Sentence:
    def __init__(self):
        self.tokens: List[ConLLToken] = []
        self.predicates: List = []
        self.predicate_base: List = []
        self.labels_list_dict = []
        self.unique_labels = set()

    def add_token(self, token: ConLLToken):
        """ Add a token to the sentence.
        

            Parameters
            ----------
            token : ConLLToken
                The token to add to the sentence.
            
            Returns
            -------
            None
        """

        self.tokens.append(token)
        self.unique_labels.update(token.labels) # Update unique labels

        if token.is_pred: # If it is a predicate add to the list
            self.predicates.append(token.token)
            self.predicate_base.append(token.pred)

    def build_labels(self):
        """ Build the labels for the sentence.

            Returns
            -------
            None 
        """
        for tok in self.tokens:
            label_dict = {}
            for i, pred in enumerate(self.predicates):
                label_dict[i] = tok.labels[i]
            self.labels_list_dict.append(label_dict)

    def to_pandas(self) -> pd.DataFrame:
        global_data_dict: Dict = {
            'id': [],
            'position': [],
            'token': [],
            'lemma': [],
            'pos_u': [],
            'pos_tag': [], 
            'd_tag': [],
            'head': [],
            'dep_tag': [],
            'is_pred': [],
            'pred': [],
            'pred_base': [],
            'label': []
        }

        # Create one instance of the sentence per predicate
        for j, pred in enumerate(self.predicates):
            data_dict: Dict = {
                'id': [],
                'position': [],
                'token': [],
                'lemma': [],
                'pos_u': [],
                'pos_tag': [],  
                'd_tag': [],
                'head': [],
                'dep_tag': [],
                'is_pred': [],
                'pred': [],
                'pred_base': [],
                'label': []
            }

            # Iterate over all the tokens 
            for i, tok in enumerate(self.tokens):
                label_dict = self.labels_list_dict[i] 

                data_dict['id'].append(tok.id)
                data_dict['position'].append(tok.position)
                data_dict['token'].append(tok.token)
                data_dict['lemma'].append(tok.lemma)
                data_dict['pos_u'].append(tok.pos_u)
                data_dict['pos_tag'].append(tok.pos_tag)
                data_dict['d_tag'].append(tok.d_tag)
                data_dict['head'].append(tok.head)
                data_dict['dep_tag'].append(tok.dep_tag)
                data_dict['is_pred'].append(tok.is_pred)
                data_dict['pred'].append(pred)
                data_dict['pred_base'].append(self.predicate_base[j])
                data_dict['label'].append(label_dict[j])

            # Append as list
            for key in data_dict.keys():
                global_data_dict[key].append(data_dict[key])

        return pd.DataFrame.from_dict(global_data_dict)

    def __str__(self):
        return str(self.tokens)
    
    def __repr__(self):
        return str(self.tokens) 

    @staticmethod
    def create_annotation(line_buffer: List[str]):
        sentence: Sentence = Sentence()
        word_idx = 0
        for i, line in enumerate(line_buffer):
            line_data = line.split('\t')
            # No problem lines with too few characters
            if len(line_data) <= 8:
                continue
            # Punctuation is not a token
            # if '.' in line_data[0] or '-' in line_data[0]:
            #     continue
            token: ConLLToken = ConLLToken(line_data, word_idx, i)
            sentence.add_token(token)
            word_idx += 1
        sentence.build_labels()
        return sentence

    @staticmethod
    def get_unique_labels(sentences: List) -> List[str]:
        labels: set = set()
        for sentence in sentences:
            labels = labels.union(sentence.unique_labels)
        return list(labels)



def parse(filename: str) -> pd.DataFrame:
    """ Parse a conllu file and return a dataframe with the parsed data
    """
    df_dict: Dict = {
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
    with open(raw_dir + filename, 'r', encoding="utf8") as f:
        sent_cnt = 0
        for line in f.readlines():
            if line.startswith('#'):
                continue
            line = line.strip()
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
                clean_args = [remove_prefix(x) for x in columns[11:]]
                df_dict['up:args'].append(clean_args)
            else:
                df_dict['up:preds'].append('_')
                df_dict['up:args'].append(['_'])
    return pd.DataFrame(df_dict)

def preprocessing(filename: str) -> pd.DataFrame:
    """  Preprocess the conllu file and return a dataframe with the preprocessed data


        Parameters
        ----------
        filename : str
            The name of the file to preprocess.
        
        Returns
        -------
        pd.DataFrame
            The preprocessed data in a pandas dataframe.
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

def preprocessing_2(filename: str) -> pd.DataFrame:
    """  Preprocess the conllu file and return a dataframe with the preprocessed data
    """
    df = parse(filename)

    df = df.rename({'up:preds': 'predicate', 'up:args': 'label'}, axis=1)

    # Get unique sentence ids
    sentence_ids = df['sentence_num'].unique()


    # Initialize columns
    df['sentence_num_clone'] = 0
    df_list = []
    
    new_sentence_id = df['sentence_num'].max() + 1

    remove_rows = []

    # For each senttence
    for sentence_id in sentence_ids:
        sentence_df = df[df['sentence_num'] == sentence_id]

        # Extract the predicates of this sentence

        predicate_dict = {

        }

        # Extract the actual tokens of the predicates
        for i, row in sentence_df.iterrows():
            if row['predicate'] == '_':
                continue
            try:
                pred_num = row['label'].index('V')
            except:
                remove_rows.append(i)
                break
            predicate_dict[pred_num] = {}
            predicate_dict[pred_num]['token'] = row['token']
            predicate_dict[pred_num]['predicate'] = row['predicate']
            predicate_dict[pred_num]['token_id'] = row['token_id']

        for i, pred_num in enumerate(predicate_dict.keys()):
            pred_dict_local = predicate_dict[pred_num]
            new_sentence_df = sentence_df.copy()
            new_sentence_df['sentence_num_clone'] = new_sentence_id if i != 0 else sentence_id
            new_sentence_df['predicate_token'] = pred_dict_local['token']
            new_sentence_df['predicate_token_id'] = pred_dict_local['token_id']
            new_sentence_df['label'] = new_sentence_df['label'].apply(lambda x: check_label(x, i))

            df_list.append(new_sentence_df)

            if i != 0:
                new_sentence_id += 1

    print(remove_rows)
    #df.drop(remove_rows, inplace=True)
    new_df = pd.concat(df_list)
    new_df['sentence_id'] = new_df['sentence_num_clone']
    new_df = new_df.drop(['sentence_num_clone'], axis=1)

    file_name = filename.split('.')[0]
    new_df.to_csv(f'data/preprocessed/{file_name}_pp.tsv', index=False, sep='\t')
    return new_df


def preprocessing_3(filename: str, force=False) -> pd.DataFrame:
    file_name = filename.split('.')[0]

    # If file is already preprocessed not force it
    if os.path.exists(f'data/preprocessed/{file_name}_pp_1.tsv') and not force:
        df = pd.read_csv(f'data/preprocessed/{file_name}_pp_1.tsv', sep='\t')
        label_list = set()
        for label in df['label']:
            label_list.update(label)
        return df, list(label_list)

    sentences: List[Sentence] = []
    buffer: List = []

    # Create list of sentences
    with open(raw_dir + filename, 'r', encoding="utf8") as f:
        for line in f.readlines():
            # Coments
            if line.startswith('#'):
                continue
            line = line.strip()
            columns = line.split('\t')

            #TODO Can this change ? 
            # Invalid lines are not added
            if len(columns) == 0 or len(line) < 5:
                try:
                    sent: Sentence = Sentence.create_annotation(buffer)
                except InvalidTokenException as e:
                    print('Skipping sentence with invalid format')
                    buffer = []
                    continue
                buffer = []
                sentences.append(sent)
                continue
            buffer.append(line)


        # Add last sentence
        if len(buffer) > 0:
            sent: Sentence = Sentence.create_annotation(buffer)
            sentences.append(sent)

    # Create the final dataframe
    df_list = [sent.to_pandas().assign(sentence_id=i) for i, sent in enumerate(sentences)]

    df = pd.concat(df_list).reset_index(drop=True)
    df.to_csv(f'data/preprocessed/{file_name}_pp_1.tsv', index=False, sep='\t')

    label_list = Sentence.get_unique_labels(sentences)
    label_list = sorted(label_list)


    return df, label_list
