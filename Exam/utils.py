import numpy as np
import re
from checklist.perturb import process_ret, Perturb

import spacy
nlp = spacy.load("en_core_web_sm")


def change_loc(doc, meta=False, seed=None, n=10):
    pred_pos = doc[1]
    arg_pos = doc[2]
    label = doc[3]
    pred_frame = doc[4]
    doc = list(nlp.pipe([doc[0]]))[0]
    ret = Perturb.change_location(doc, meta, n=n, seed=seed)
    data_tup = [(x, pred_pos, arg_pos, label, pred_frame) for x in ret]
    return  data_tup



def perturb_time_past(doc, meta=False, seed=None, n=10):
    pred_pos = doc[1]
    arg_pos = doc[2]
    label = doc[3]
    pred_frame = doc[4]
    doc = list(nlp.pipe([doc[0]]))[0]
    if seed is not None:
        np.random.seed(seed)
    past = ['last week', 'last month', 'yesterday', 'this morning', 'last night', 'last year', 'recently']

    t_location = [x.text for x in doc if x.text.lower() in past]

    ret = []
    ret_m = []
    for x in t_location:
        sub_re = re.compile(r'\b%s\b' % x)
        ret.extend([sub_re.sub(n, doc.text) for n in past])
        ret_m.extend([(x, n) for n in past])
    data_tup = [(x, pred_pos, arg_pos, label, pred_frame) for x in ret]
    return data_tup

def perturb_time_present(doc, meta=False, seed=None, n=10):
    pred_pos = doc[1]
    arg_pos = doc[2]
    label = doc[3]
    pred_frame = doc[4]
    doc = list(nlp.pipe([doc[0]]))[0]

    if seed is not None:
        np.random.seed(seed)
    present = ['tomorrow', 'today', 'this afternoon', 'tonight']

    t_location = [x.text for x in doc if x.text.lower() in present]

    ret = []
    ret_m = []
    for x in t_location:
        sub_re = re.compile(r'\b%s\b' % x)
        ret.extend([sub_re.sub(n, doc.text) for n in present])
        ret_m.extend([(x, n) for n in present])
    data_tup = [(x, pred_pos, arg_pos, label, pred_frame) for x in ret]
    return data_tup

def perturb_time_future(doc, meta=False, seed=None, n=10):
    pred_pos = doc[1]
    arg_pos = doc[2]
    label = doc[3]
    pred_frame = doc[4]
    doc = list(nlp.pipe([doc[0]]))[0]
    if seed is not None:
        np.random.seed(seed)
    future = ['next week', 'next month', 'next year', 'tomorrow', 'tonight']

    t_location = [x.text for x in doc if x.text.lower() in future]

    ret = []
    ret_m = []
    for x in t_location:
        sub_re = re.compile(r'\b%s\b' % x)
        ret.extend([sub_re.sub(n, doc.text) for n in future])
        ret_m.extend([(x, n) for n in future])
    data_tup = [(x, pred_pos, arg_pos, label, pred_frame) for x in ret]
    return data_tup

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
    tok_sent = tokenizer(row["word"], is_split_into_words=True)
    tok_whole = tokenizer(row["word"], [pred_token], padding='max_length', max_length=64, truncation=True, is_split_into_words=True)

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

def perturb_date_ref(doc, meta=False, seed=None, n=10):
    """Changes the time reference

    Parameters
    ----------
    doc : spacy.token.Doc
        input
    meta : bool
        if True, will return list of (orig_t_ref, new_t_ref) as meta
    seed : int
        random seed
    n : int
        number of temporal locations to replace original locations with

    Returns
    -------
    list(str)
        if meta=True, returns (list(str), list(tuple))
        Strings with numbers replaced.

    """
    pred_pos = doc[1]
    arg_pos = doc[2]
    label = doc[3]
    pred_frame = doc[4]
    doc = list(nlp.pipe([doc[0]]))[0]
    if seed is not None:
        np.random.seed(seed)
    dow = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    c_dow = [d.capitalize() for d in dow]
    c_months = [m.capitalize() for m in months]
    t_location = [x.text for x in doc if x.text.lower() in dow + months]
    ret = []
    ret_m = []
    for x in t_location:
        #x = x.lower()
        sub_re = re.compile(r'\b%s\b' % x)
        to_sub = c_months+c_dow
        ret.extend([sub_re.sub(n, doc.text) for n in to_sub])
        ret_m.extend([(x, n) for n in to_sub])
    data_tup = [(x, pred_pos, arg_pos, label, pred_frame) for x in ret]
    return data_tup

def tokenize_and_align_labels(tokenizer, row):
    """ Tokenize the inputs and align the labels with them. It is particularly important to
        note the step in adding the tokenized predicate at the end of the input embeddings.

   Args:
        tokenizer: Tokenizer
        row: dict
    Returns:
        dict

    """
    words = row['sent'].split(' ') if type(row['sent']) is not list else row['sent']
    pred_token = row['pred']
    tok_sent = tokenizer(words, is_split_into_words=True)
    tok_whole = tokenizer(words, [pred_token], padding='max_length', max_length=64, truncation=True, is_split_into_words=True)

    label_ids = []

    for i, word_idx in enumerate(tok_whole.word_ids()):
        if word_idx is None:
            # If it is a special token do not add a label
            label_ids.append(-100)
        elif i >= len(tok_sent['input_ids']):
            # If the token is part of the predicate do not add a label
            label_ids.append(-100)
        elif 'arg_pos' in row and word_idx == row['arg_pos']:
            label_ids.append(row['label'])
        else: 
            label_ids.append(-100)

    tok_whole["labels"] = label_ids
    tok_whole['sent'] = row['sent']
    return tok_whole 