import pandas as pd

def generate_ngrams(df):
    
    fw_bigrams = []
    bw_bigrams = []
    fw_trigrams = []
    bw_trigrams = []

    fw_pos_bigrams = []
    bw_pos_bigrams = []
    fw_pos_trigrams = []
    bw_pos_trigrams = []

    max_sentence_id = max(df['sentence_id'])
    
    for id in range(1, max_sentence_id + 1):
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
        fw_trigram_shift = sentence['word'].shift(-2)
        fw_trigram = (sentence['token'] + ' ' + fw_bigram_shift.fillna('') + ' ' + fw_trigram_shift.fillna('')).tolist()
        fw_trigrams.extend(fw_trigram)

        # Backward tokentrigram
        bw_trigram = (sentence['token'].shift(2).fillna('') + ' ' + sentence['token'].shift(1).fillna('') + ' ' + sentence['token']).tolist()
        bw_trigrams.extend(bw_trigram)

        # POS N-grams:

        # Forward POS bigram
        fw_pos_bigram = (sentence['POS'] + ' ' + sentence['POS'].shift(-1).fillna('')).tolist()
        fw_pos_bigrams.extend(fw_pos_bigram)

        # Backward POS bigram
        bw_pos_bigram = (sentence['POS'].shift(1).fillna('') + ' ' + sentence['POS']).tolist()
        bw_pos_bigrams.extend(bw_pos_bigram)

        # Forward POS trigram
        fw_pos_trigram = (sentence['POS'] + ' ' + sentence['POS'].shift(-1).fillna('') + ' ' + sentence['POS'].shift(-2).fillna('')).tolist()
        fw_pos_trigrams.extend(fw_pos_trigram)

        # Backward POS trigram
        bw_pos_trigram = (sentence['POS'].shift(2).fillna('') + ' ' + sentence['POS'].shift(1).fillna('') + ' ' + sentence['POS']).tolist()
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


if __name__ == '__main__':

    df = pd.read_csv('A2/preprocessed_train_with_header.tsv', sep='\t', header=0)
    df_new = generate_ngrams(df)
    df_new.to_csv('A2/data/feature_extraction_nur.csv', index=False) # to test the implementation