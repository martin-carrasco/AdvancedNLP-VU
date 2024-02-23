import spacy
import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('data/dev_2.csv')
    df_ner: pd.DataFrame = ner_tagger(df)
    df_ngram = generate_ngrams(df_ner)
     
    df_ngram.to_csv('data/dev_2_final') # to test the implementation