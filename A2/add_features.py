import pandas as pd
import spacy
import tqdm
import benepar

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})


def add_features_1(df: pd.DataFrame) -> pd.DataFrame:
    # Initialize columns as 0 
    df['position_rel_left']  = 0
    df['position_rel_right']  = 0

    # Phrase label of the token or parent
    df['phrase_type']  = ''

    # Active voice or passive voice depending on the dependency relation
    df['voice_passive'] = df.apply(lambda x: True if 'nsubj' in x['deprel'] else False, axis=1)
    df['voice_active'] = df.apply(lambda x: True if 'nsubjpass' in x['deprel'] else False, axis=1)

    # Iterate over each sentence
    for sentence_id in tqdm.tqdm(list(df['sentence_id'].unique())):
        sentence_df = df[df['sentence_id'] == sentence_id]

        sentence_text = ''
        # Add space after words if required
        for i, row in sentence_df.iterrows():
            sentence_text += row['token']
            if 'SpaceAfter' not in row['misc']:
                sentence_text += ' '
        
        # Iterate over each token in the sentence
        for token_id in list(sentence_df['token_id'].unique()):
            # Fetch the token row
            token_df = df[(df['token_id'] == token_id) & (df['sentence_id'] == sentence_id)]
            token = token_df['token']
            
            # Extract the predicate token id 
            predicate =  token_df['predicate_token_id']

            # Extract the relative position of the token in the sentence
            df.loc[((df['sentence_id'] == sentence_id) & (df['token_id'] == token_id)), 'position_rel_left'] =  token_id < predicate
            df.loc[((df['sentence_id'] == sentence_id) & (df['token_id'] == token_id)), 'position_rel_right'] =  token_id > predicate


        doc = nlp(sentence_text)

        # Iterate over each node in the constituency tree
        for t_id, token in enumerate(list(doc.sents)[0]):
            token_id = t_id + 1
            phrase_type = ''
            current_span = token
            # Iterate over the parent of the token to find the phrase type
            # of the closest ancestor 
            while phrase_type == '':
                if not len(current_span._.labels):
                    current_span = current_span._.parent
                else:
                    phrase_type = current_span._.labels[0]
                    break
            mask = (df['sentence_id'] == sentence_id) & (df['token_id'] == token_id)
            df.loc[mask, 'phrase_type'] = phrase_type
    return df

def add_features_2(df: pd.DataFrame) -> pd.DataFrame:
    """ Generate n-grams for the given dataframe
    """
    
    fw_bigrams = []
    bw_bigrams = []
    fw_trigrams = []
    bw_trigrams = []

    fw_pos_bigrams = []
    bw_pos_bigrams = []
    fw_pos_trigrams = []
    bw_pos_trigrams = []

    max_sentence_id = max(df['sentence_id'])
    
    for id in range(0, max_sentence_id + 1):
        sentence = df.loc[df['sentence_id'] == id]

        # Token N-grams:
        
        # Forward token bigram
        fw_bigram_shift = sentence['token'].shift(-1)
        fw_bigram = (sentence['token'] + ' ' + fw_bigram_shift.fillna('')).tolist()
        fw_bigrams.extend(fw_bigram)

        # Backward token bigram
        bw_bigram_shift = sentence['token'].shift(1)
        bw_bigram = (bw_bigram_shift.fillna('') + ' ' + sentence['token']).tolist()
        bw_bigrams.extend(bw_bigram)

        # Forward token trigram
        fw_trigram_shift = sentence['token'].shift(-2)
        fw_trigram = (sentence['token'] + ' ' + fw_bigram_shift.fillna('') + ' ' + fw_trigram_shift.fillna('')).tolist()
        fw_trigrams.extend(fw_trigram)

        # Backward token trigram
        bw_trigram_shift = sentence['token'].shift(2)
        bw_trigram = (bw_trigram_shift.fillna('') + ' ' + bw_bigram_shift.fillna('') + ' ' + sentence['token']).tolist()
        bw_trigrams.extend(bw_trigram)

        # POS N-grams:

        # Forward POS bigram
        fw_pos_bigram_shift = sentence['POS'].shift(-1)
        fw_pos_bigram = (sentence['POS'] + ' ' + fw_pos_bigram_shift.fillna('')).tolist()
        fw_pos_bigrams.extend(fw_pos_bigram)

        # Backward POS bigram
        bw_pos_bigram_shift = sentence['POS'].shift(1)
        bw_pos_bigram = (bw_pos_bigram_shift.fillna('') + ' ' + sentence['POS']).tolist()
        bw_pos_bigrams.extend(bw_pos_bigram)

        # Forward POS trigram
        fw_pos_trigram_shift = sentence['POS'].shift(-2)
        fw_pos_trigram = (sentence['POS'] + ' ' + fw_pos_bigram_shift.fillna('') + ' ' + fw_pos_trigram_shift.fillna('')).tolist()
        fw_pos_trigrams.extend(fw_pos_trigram)

        # Backward POS trigram
        bw_pos_trigram_shift = sentence['POS'].shift(2)
        bw_pos_trigram = (bw_pos_trigram_shift.fillna('') + ' ' + bw_pos_bigram_shift.fillna('') + ' ' + sentence['POS']).tolist()
        bw_pos_trigrams.extend(bw_pos_trigram)

    df['fw_bigram'] = fw_bigrams
    df['bw_bigram'] = bw_bigrams
    df['fw_trigram'] = fw_trigrams
    df['bw_trigram'] = bw_trigrams
    df['fw_pos_bigram'] = fw_pos_bigrams
    df['bw_pos_bigram'] = bw_pos_bigrams
    df['fw_pos_trigram'] = fw_pos_trigrams
    df['bw_pos_trigram'] = bw_pos_trigrams

    return df


def add_features_3(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Add a 'named_entity' column to the DataFrame by performing Named Entity Recognition using spaCy.
    Args: df (pandas.DataFrame): DataFrame with a 'token' column containing tokenized text
    Returns: pandas.DataFrame: DataFrame with an additional 'named_entity' column containing named entity tags
    """
    
    # Load English tokenizer, tagger, parser, NER
    nlp = spacy.load("en_core_web_md")
    
    # Process each token and obtain named entity tags
    named_entity_tags = []
    for token in df['token']:
        if isinstance(token, str):
            doc = nlp(token)
            named_entity_tags.append(doc.ents[0].label_ if doc.ents else "O")  # Using the label of the first entity if present, else 'O'
        else:
            named_entity_tags.append("O")  # Handle NaN values
    
    # Add named entity tags as a new column in the DataFrame
    df['NER_tag'] = named_entity_tags
    
    return df
