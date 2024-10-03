from typing import List, Dict
import pandas as pd
from A3.preproc.utils import format_label

class InvalidTokenException(Exception):
    pass

class ConLLToken:
    def __init__(self, data: List, element_idx, e_id):
        self.data = data # Save original data

        try:
            self.id = int(data[0])
        except ValueError as e: 
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
            self.labels = [format_label(p) for p in data[11:]]

    def __str__(self):
        return self.token
    def __repr__(self):
        return self.token

class Sentence:
    # Fields that will be fetched from ConLLToken
    VALID_FIELDS = [
        'id',
        'position',
        'token',
        'lemma',
        'pos_u',
        'pos_tag', 
        'd_tag',
        'head',
        'dep_tag',
        'is_pred',
        'pred',
        'pred_base',
        'label'
    ]
    def __init__(self):
        self.tokens: List[ConLLToken] = []
        self.predicates: List = []
        self.predicate_base: List = []
        # This will have the same size as self.tokens
        # each i-th element represents a dict of (idx, label)
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

    def to_pandas(self) -> pd.DataFrame:
        # Each field is an empty list
        dataframe_dict: Dict = {
            f'{field}': [] for field in self.VALID_FIELDS
        }

        # Create one instance of the sentence per predicate
        for j, preds in enumerate(zip(self.predicates, self.predicate_base)):
            pred, pred_base = preds
            current_dict = {
                f'{field}': [] for field in self.VALID_FIELDS
            }

            # Iterate over all the tokens 
            for i, tok in enumerate(self.tokens):
                # For each of the valid fields we extract
                for field in self.VALID_FIELDS:
                    if field == 'pred':
                        current_dict[field].append(pred)
                    elif field == 'pred_base':
                        current_dict[field].append(pred_base)
                    elif field == 'label':
                        current_dict[field].append(getattr(tok, 'labels')[j])
                    else:
                        current_dict[field].append(getattr(tok, field))

            # Append as list
            for key in current_dict.keys():
                dataframe_dict[key].append(current_dict[key])

        return pd.DataFrame.from_dict(dataframe_dict)

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
            token: ConLLToken = ConLLToken(line_data, word_idx, i)
            sentence.add_token(token)
            word_idx += 1
        return sentence

    @staticmethod
    def get_unique_labels(sentences: List) -> List[str]:
        labels: set = set()
        for sentence in sentences:
            labels = labels.union(sentence.unique_labels)
        return list(labels)
