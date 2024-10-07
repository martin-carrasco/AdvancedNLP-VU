from typing import List, Dict
import os
import pandas as pd
import ast

from preproc.structures import Sentence, InvalidTokenException

RAW_DIR = 'data/raw/'

def preprocess(filename: str, force=True) -> pd.DataFrame:
    file_name = filename.split('.')[0]

    # If file is already preprocessed not force it
    if os.path.exists(f'data/preprocessed/{file_name}_pp.tsv') and not force:
        df = pd.read_csv(f'data/preprocessed/{file_name}_pp.tsv', sep='\t')
        label_list = set()
        for label in df['label']:
            label_lst = ast.literal_eval(label)
            label_list.update(label_lst)
        return df, list(label_list)

    sentences: List[Sentence] = []
    buffer: List = []

    # Create list of sentences
    with open(RAW_DIR + filename, 'r', encoding="utf8") as f:
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
    df.to_csv(f'data/preprocessed/{file_name}_pp.tsv', index=False, sep='\t')

    label_list = Sentence.get_unique_labels(sentences)
    label_list = sorted(label_list)


    return df, label_list