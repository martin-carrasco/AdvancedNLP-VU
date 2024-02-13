import spacy
import benepar
import nltk

def sbar(head):
    """ Find a node possibly marked as SBAR
    """
    if not head.children:
        return None
    for node in head.children:
        pass

nlp = spacy.load('en_core_web_md')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
doc = nlp('Those who knew David.')

sent = list(doc.sents)[0]
print(sent._.parse_string)
# (S (NP (NP (DT The) (NN time)) (PP (IN for) (NP (NN action)))) (VP (VBZ is) (ADVP (RB now))) (. .))
print(sent._.labels)
# ('S',)
print(list(sent._.children)[0])
