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

if __name__ == '__main__':
    
    preprocessing('A2/data/en_ewt-up-train.conllu', 'A2/data/en_ewt-up-train_new.conllu')