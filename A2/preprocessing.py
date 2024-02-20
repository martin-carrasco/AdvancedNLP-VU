import pandas as pd

def preprocessing(input_filepath, output_filepath):
    sentence_id = 1  # Initialize sentence ID counter

    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            line = line.strip()  # Remove leading/trailing whitespaces

            # Skip lines that are either empty or start with '#'
            if line and not line.startswith('#'):
                # Write the line along with the sentence ID to the output file
                outfile.write(f'{sentence_id}\t{line}\n')
            else:
                # If the line is empty, it marks the end of a sentence
                if not line:
                    sentence_id += 1  # Increment sentence ID for the next sentence

    with open(output_filepath, encoding='utf-8') as file:
        lines = file.readlines()

    data = [line.strip().split('\t') for line in lines]
    max_columns = max(len(row) for row in data)
    print(max_columns)
    new_df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(max_columns)])
    
    # Rename only the first 11 columns (sentence_num + 10 other - see assignment guidelines)
    additional_cols = [f'col_{i}' for i in range(11, max_columns)]
    new_df.columns = ['sentence_num', 'token', 'lemma', 'UPOS', 'POS', 'grammar', 'head_id', 'dependency_label',
                      'head_dependency_relation', 'additional_info', 'is_predicate'] + additional_cols

    ##################################################################################################################
    # TODO: To my understanding, all of these additional columns means there are sentences with up to 35 predicates. #
    # We need to duplicate the sentences for each predicate and then drop the column with no unique name.            #
    ##################################################################################################################
    
    '''# Get a list of column names starting with 'col_' -> didn't work as intended
    columns_to_drop = []

    # Iterate through columns
    for column in new_df.columns:
        if column.startswith('col_'):
            # Check if any non-empty value exists in the column
            if new_df[column].notna().any():
                continue  # Keep the column
            else:
                columns_to_drop.append(column)  # Mark for dropping'''
    
    return new_df        

if __name__ == '__main__':
    
    df = preprocessing('A2/data/en_ewt-up-train.conllu', 'A2/data/en_ewt-up-train_new.conllu')
    df.to_csv('A2/data/en_ewt-up-train_new.csv', index=False)