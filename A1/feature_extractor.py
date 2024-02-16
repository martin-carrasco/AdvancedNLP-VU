import spacy
import pandas as pd
import benepar
import nltk
from nltk import Tree
from typing import List, Tuple
from nltk.corpus import gutenberg, genesis

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def leading_advb(head):
    """  The index of the span containing a leading Adverbial Phrase
    """
    if not list(head._.children):
        return []
    span_idx: List[Tuple] = []
    if len(list(head._.children)[0]._.labels) and 'ADVP' in list(head._.children)[0]._.labels[0]:
        span_idx.append((list(head._.children)[0].start_char, list(head._.children)[0].end_char))
    else:
        for child in head._.children:
            span_idx += leading_advb(child)
    return span_idx
    
def find_gvb_as_np(head, is_np=False):
    """ The index of the span containing a Verb Gerundive Phrase as 
    part of a Noun Phrase
    """
    # No children so no point in checking
    if not list(head._.children):
        return []
    span_idx: List[Tuple] = []
    is_np = True if 'NP' in head._.labels else is_np
    if 'VP' in head._.labels and is_np:
        for child in head._.children:
            if 'VBG' in child._.labels or 'VBG' in child._.parse_string:
                span_idx.append((head.start_char, head.end_char))
                break
    else:
        for child in head._.children:
            span_idx += find_gvb_as_np(child, is_np)
    return span_idx
        
def find_parenthetical(head):
    """ The index of the span containing a Parenthetical
    """
    # No children so no point in checking
    if not list(head._.children):
        return [] 
    # List of tuples (i, j) of index
    span_idx: List[Tuple] = []
    # Since it's not an NP, we need to check the children to see 
    # if we get to onne
    if 'NP' not in head._.labels:
        # Recursively check the children
        for child in head._.children:
            span_idx = span_idx + find_parenthetical(child)
    # Its an NP so then we check if it is possible that 
    # it contains a Parenthetical
    else: 
        for i, node in enumerate(head._.children):
            if 'PRN' in node._.labels:
                span_idx.append((node.start_char, node.end_char))
    return span_idx 

def find_npa(head):
    """ The index of the span containing a Noun Phrase Appositive
    """
    # No children so no point in checking
    if not head._.children:
        return [] 
    # List of tuples (i, j) of index
    npa_idx: List[Tuple] = []
    # Since it's not an NP, we need to check the children to see 
    # if we get to onne
    if 'NP' not in head._.labels:
        # Recursively check the children
        for child in head._.children:
            npa_idx = npa_idx + find_npa(child)
    # Its an NP so then we check if it is possible that 
    # its formed by NPs in which one is an appositive
    else: 
        for i, node in enumerate(head._.children):
            # First node cannot be appostiive
            if i == 0:
                continue
            if 'NP' in node._.labels:
                if list(head._.children)[i-1].text == ',':
                    npa_idx.append((node.start_char, node.end_char))
                    break
    return npa_idx


def parse_sentences(lines: List):
    df_dict = {
        'is_npa': [],
        'is_parenthetical': [],
        'is_gvb_as_np': [],
        'is_leading_advb': [],
        'token_id': [],
        'sentence_num': [],
        'word': [],
    }
    for sent_id, sent in enumerate(lines):
        doc = nlp(sent)

        npa_span_idx = find_npa(list(doc.sents)[0])
        parenthetical_span_idx = find_parenthetical(list(doc.sents)[0])
        gvb_as_np_span_idx = find_gvb_as_np(list(doc.sents)[0])
        leading_advb_span_idx = leading_advb(list(doc.sents)[0])

        for token_id, token in enumerate(list(doc.sents)[0]):
            is_npa = False
            is_parenthetical = False
            is_gvb_as_np = False
            is_leading_advb = False

            start_i = token.idx
            end_i = start_i + len(token.text)

            for span_i in npa_span_idx:
                if start_i >= span_i[0] and end_i <= span_i[1]:
                    is_npa = True
                    break
            for span_i in parenthetical_span_idx:
                if start_i >= span_i[0] and end_i <= span_i[1]:
                    is_parenthetical = True
                    break
            for span_i in gvb_as_np_span_idx:
                if start_i >= span_i[0] and end_i <= span_i[1]:
                    is_gvb_as_np = True
                    break
            for span_i in leading_advb_span_idx:
                if start_i >= span_i[0] and end_i <= span_i[1]:
                    is_leading_advb = True
                    break

            df_dict['is_npa'].append(is_npa)
            df_dict['is_parenthetical'].append(is_parenthetical)
            df_dict['is_gvb_as_np'].append(is_gvb_as_np)
            df_dict['is_leading_advb'].append(is_leading_advb)

            df_dict['token_id'].append(token_id)
            df_dict['sentence_num'].append(sent_id)
            df_dict['word'].append(token.text)

        print(df_dict)
        exit()


        

def load_data(datafile):
    new_lines = []
    for gutenberg_file in gutenberg.fileids():
        lines = gutenberg.sents(gutenberg_file)
        for line in lines:
            line = ' '.join(line)
            new_lines.append(line)
        break
    return new_lines
#lines = load_data('data/blue_castle_raw.txt')
lines = ['Allen Ginsberg , a famous poet, wrote "Howl", a controversial poem.']


df = parse_sentences(lines)

print(df['sentence_num'])
