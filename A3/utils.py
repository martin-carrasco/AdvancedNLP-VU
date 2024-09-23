import numpy as np
from A3.preprocessing import preprocessing, preprocessing_3
from datasets import Dataset, load_metric, load_dataset, Sequence, ClassLabel, Features, Value

metric = load_metric("seqeval")

label_all_tokens = False

def process_df_into_ds(filename: str):
    """ Load the dataset and process it into a format that can be used by the model.
        Args:
            filename: str - the name of the file to be loaded
        Returns:
            ds: dataset - the processed dataset
    """
    df, label_list = preprocessing_3(filename, force=True)

    features = Features({
        'id': Sequence(feature=Value('float32')),
        'position': Sequence(feature=Value('float32')),
        'token': Sequence(feature=Value('string')),
        'lemma': Sequence(feature=Value('string')),
        'pos_u': Sequence(feature=Value('string')),
        'pos_tag': Sequence(feature=Value('string')),
        'd_tag': Sequence(feature=Value('string')),
        'head': Sequence(feature=Value('string')),
        'dep_tag': Sequence(feature=Value('string')),
        'is_pred': Sequence(feature=Value('bool')),
        'pred': Sequence(feature=Value('string')),
        'pred_base': Sequence(feature=Value('string')),
        'label': Sequence(feature=ClassLabel(names=label_list)),
        'sentence_id': Value('int32'),

    })

    ds = Dataset.from_pandas(df[list(features.keys())], features=features)
    return ds


def tokenize_and_align_labels_2(tokenizer, row):
    """ Tokenize the inputs and align the labels with them. It is particularly important to
        note the step in adding the tokenized predicate at the end of the input embeddings.

   Args:
        tokenizer: Tokenizer
        row: dict
    Returns:
        dict

    """
    pred_token = row['pred'][0]
    pred_token_base = row['pred_base'][0]
    tok_sent = tokenizer(row["token"], is_split_into_words=True)
    tok_whole = tokenizer(row["token"], [pred_token, pred_token_base], padding='max_length', max_length=64, truncation=True, is_split_into_words=True)

    label_ids = []

    for i, word_idx in enumerate(tok_whole.word_ids()):
        if word_idx is None:
            # If it is a special token do not add a label
            label_ids.append(-100)
        elif i >= len(tok_sent['input_ids']):
            # If the token is part of the predicate do not add a label
            label_ids.append(-100)
        else: 
            # Set the label of the first token of each word
            label_ids.append(row['label'][word_idx])


    tok_whole["labels"] = label_ids
    return tok_whole 

def tokenize_and_align_labels(tokenizer, row):
    """ Tokenize the inputs and align the labels with them. It is particularly important to
        note the step in adding the tokenized predicate at the end of the input embeddings.

   Args:
        tokenizer: Tokenizer
        row: dict
    Returns:
        dict

    """
    pred_token = row['pred'][0]
    tok_sent = tokenizer(row["token"], is_split_into_words=True)
    tok_whole = tokenizer(row["token"], [pred_token], padding='max_length', max_length=64, truncation=True, is_split_into_words=True)

    label_ids = []

    for i, word_idx in enumerate(tok_whole.word_ids()):
        if word_idx is None:
            # If it is a special token do not add a label
            label_ids.append(-100)
        elif i >= len(tok_sent['input_ids']):
            # If the token is part of the predicate do not add a label
            label_ids.append(-100)
        else: 
            # Set the label of the first token of each word
            label_ids.append(row['label'][word_idx])


    tok_whole["labels"] = label_ids
    return tok_whole 

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
