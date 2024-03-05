import numpy as np
from datasets import load_dataset, load_metric

metric = load_metric("seqeval")

def check_label(x, i: int) -> str:
    if len(x) < i+1:
        return 'O'
    else:
        return x[i] if x[i] != '_' else 'O'

label_all_tokens = False

def tokenize_and_align_labels(tokenizer, row):
    tokenized_inputs = tokenizer(row["token"], padding='max_length', max_length=85, truncation=True, is_split_into_words=True)
    pred_tokens = tokenizer([row['predicate_token']], is_split_into_words=True)
    for idx in range(len(pred_tokens["input_ids"])-1):
        idx = idx+1
        tokenized_inputs['input_ids'].append(pred_tokens["input_ids"][idx])
        tokenized_inputs['token_type_ids'].append(pred_tokens["token_type_ids"][idx])
        tokenized_inputs['attention_mask'].append(pred_tokens["attention_mask"][idx])

    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        # Special tokens have a word id that is None. We set the label to -100 so they are automatically
        # ignored in the loss function.
        if word_idx is None:
            label_ids.append(-100)
        # We set the label for the first token of each word.
        elif word_idx != previous_word_idx:
            label_ids.append(row['label'][word_idx])
        # For the other tokens in a word, we set the label to either the current label or -100, depending on
        # the label_all_tokens flag.
        else:
            label_ids.append(row['label'][word_idx] if label_all_tokens else -100)
        previous_word_idx = word_idx


    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

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
