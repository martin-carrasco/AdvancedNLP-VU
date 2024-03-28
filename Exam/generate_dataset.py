import random
import json
import pipelines
import itertools
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import numpy as np

import checklist
from checklist.editor import Editor
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb
from checklist.test_types import MFT, INV, DIR
from checklist.pred_wrapper import PredictorWrapper
from checklist.expect import Expect
from pattern.en import sentiment
from utils import perturb_date_ref, perturb_time_future, perturb_time_past, perturb_time_present, change_loc


editor = Editor()

def change_pred(origin_pred, pred, origin_conf, conf, labels=None, meta=None):
    return pred[labels] == origin_pred[labels]

def add_goal(x, nsamples=10):
    phrases = ['to the university', 'to the beach', 'to the store', 'to the building', 'to the garbe disposal',
               'to the park', 'to the gym', 'to the library', 'to the hospital', 'to the school',
               'to the restaurant', 'to the cinema', 'to the museum', 'to the zoo', 'to the aquarium',
               'to the hospital', 'to the school', 'to the restaurant', 'to the cinema', 'to the museum', 'to the zoo', 'to the aquarium']
    cnt = 0
    rets = []
    while cnt < nsamples:
        x_c = random.choice(x)
        pred_pos = x_c[1]
        arg_pos = x_c[2]
        label = x_c[3]
        pred_frame = x_c[4]
        x_c = x_c[0]
        p = random.choice(phrases)
        data_tup = ('{} {}'.format(x_c, p), pred_pos, arg_pos, label, pred_frame)
        rets.append(data_tup)
        cnt+=1

    return rets

def add_source(x, nsamples=10):
    phrases = ['from the store', 'from my window', 'from my neighbourhood', 'from jail']
    cnt = 0
    rets = []
    while cnt < nsamples:
        x_c = random.choice(x)
        pred_pos = x_c[1]
        arg_pos = x_c[2]
        label = x_c[3]
        pred_frame = x_c[4]
        x_c = x_c[0]
        p = random.choice(phrases)
        data_tup = ('{} {}'.format(x_c, p), pred_pos, arg_pos, label, pred_frame)
        rets.append(data_tup)
        cnt+=1

    return rets


def test_1_CA0_R():
    # Generated using templates
    dataset = [
        ('There was a book', 'be.02'),
        ('There lived a person', 'live.01'),
        ('There lied a book', 'lie.07'),
        ('There existed a chair', 'exist.01'),
        ('There lied a phone', 'lie.07'),
        ('There died a person', 'die.01'),
        ('There exists a building', 'exist.01'),
        ('There was a phone', 'be.02'),
        ('There lays a screen', 'lie.07'),
        ('There remains a person', 'remain.01'),
        ('There lays a book',  'lie.07'),
        ('There hangs a TV', 'hang.01'),
        ('Here exists a book', 'exist.01'),
        ('Here runs a computer', 'run.01'),
        ('Here exists a phone', 'exist.01'),
        ('Here swings a chair', 'swing.02'),
        ('Here spawns a car', 'spawn.01'),
        ('Here exists a building', 'exist.01'),
        ('Here comes the sun', 'come.01'),
        ('Here projects a shadow', 'project.01'),
        ('Here was a building', 'be.02'),
        ('Here is a phone', 'be.02'),
        ('Here lays a car', 'lie.07'),
        ('Here grows a tree', 'grow.03'),
        ('Here floats a computer', 'float.01'),
    ]
    pred_pos = 1
    arg_pos = 3
    label='ARG1'

    data_tup = [(x[0], pred_pos, arg_pos, label, x[1]) for x in dataset]

    test = MFT(data=data_tup, labels=arg_pos, name='CA0-R - T1', capability='CA0-R', description='X')

    return test

def test_1_CA1_D():
    adjs = ['well', 'badly', 'quietly']
    prop = ['on', 'in', 'under', 'beside']
    nouns = ['person', 'child', 'president', 'doctor', 'cat', 'teacher']
    data = editor.template('The {nouns} {prop} {mask} {mask} is {adjs}', prop=prop, nouns=nouns, adjs=adjs, save=True, labels=5, nsamples=25)

    pred_pos = 5
    arg_pos = 1
    label='ARG0'

    data_tup = [(x, pred_pos, arg_pos, label, 'be.02') for x in data.data]

    test = MFT(data=data_tup, labels=arg_pos, name='CA1-D - T1', capability='CA1-D', description='X')
    return test

def test_1_CA0_D():
    ors = ['or', 'and', 'but', 'yet']
    verbs = ['uses', 'eats', 'drinks', 'watches', 'cries']
    verbs_frame =['use.01', 'eat.01', 'drink.01', 'watch.01', 'cry.01']
    en = ['that', 'which', 'who', 'whom']
    prep = ['Everybody', 'Nobody', 'Somebody', 'Anybody']
    adverb_a = ['swiftly', 'slowsly', 'inaudibly', 'inadvertedly', 'unintentionally']
    adverb_b = ['excitedly', 'happily', 'sadly', 'angrily', 'calmly']
    data = editor.template('{prep} {en} {adverb_a} {ors} {adverb_b} {verbs} {mask}', ors=ors, adverb_a=adverb_a, adverb_b=adverb_b, en=en, verbs=verbs, prep=prep, nsamples=25)

    pred_pos = 5
    arg_pos = 1
    label = 'ARG1'

    data_tup = []
    for x in data.data:
        idx = verbs.index(x.split(' ')[5])
        data_tup.append((x, pred_pos, arg_pos, label, verbs_frame[idx]))

    test = MFT(data=data_tup, labels=arg_pos, name='CA0-D - T1', capability='CA0-D', description='X')
    return test
    
def test_1_CA_LOC_L():
    label = 'ARGM-LOC'
    dataset = [
        ('I bought flowers in Miami', 1, 4, label, 'buy.01'),
        ('My mother used to live in Mexico', 4, 6, label, 'live.01'),
        ('There is a fantastic flower shop in China', 1, 7, label, 'be.02')
    ]
    t = Perturb.perturb(dataset, change_loc, nsamples=25, keep_original=False)
    data_tup = t.data

    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-LOC-L - T1', capability='CA-LOC-L', description='X')
    return test

def test_1_CA_NEG_A():
    na = ['never', 'hardly', 'rarely', 'seldom', 'barely']
    sents = [
        ('I {na} go to the gym', 'go.01'),
        ('I {na} eat meat', 'eat.01'),
        ('I {na} drink alcohol', 'drink.01'),
        ('I {na} watch TV', 'watch.01'),
        ('I {na} play video games', 'play.01'),
        ('I {na} read books', 'read.01'),
        ('He {na} goes to the gym', 'go.01'),
        ('He {na} eats meat', 'eat.01'),
        ('He {na} drinks alcohol', 'drink.01'),
        ('He {na} watches TV', 'watch.01')
    ]
    pred_pos = 2
    arg_pos = 1
    label = 'ARGM-NEG'

    data_tup = []
    for n_a in na:
        for s in sents:
            sent =  s[0].format(na=n_a) 
            data_tup.append((sent, pred_pos, arg_pos, label, s[1]))
    
    test = MFT(data=data_tup, labels=arg_pos, name='CA-NEG-A - T1', capability='CA-NEG-A', description='X')
    return test

def test_1_CA_TMP_D():
    label = 'ARGM-TMP'
    dataset = [
        ('My brithday is on Monday', 2, 4, label, 'be.02'),
        ('I visited my grandmother last May', 2, 5, label, 'visit.01'),
        ('I will have a party next Saturday', 2 , 6, label, 'have.04')
    ]

    t = Perturb.perturb(dataset, perturb_date_ref, nsamples=50)
    data_tup_raw = t.data
    data_tup = []
    for x in data_tup_raw:
        for y in x:
            data_tup.append((y[0], y[1], y[2], label, x[4]))

    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-TMP-D - T1', capability='CA-TMP-D', description='X')
    return test

def test_1_CA_TMP_F():
    label = 'ARGM-TMP'
    sentences_past = [
        ('I visited my grandmother recently', 2, 4, label, 'visit.01'),
        ('I ate a cake yesterday', 2, 4, label, 'eat.01'),
        ('He slept last night', 2, 3, label, 'sleep.01'),
        ('They went to the beach last week', 2, 6, label, 'go.01'),
        ('She cared for her dog last month', 2, 6, label, 'care.03')
    ]

    t = Perturb.perturb(sentences_past, perturb_time_past, nsamples=50)
    data_tup = t.data
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-TMP-F - T1', capability='CA-TMP-F', description='X')
    return test

def test_2_CA_TMP_F():
    label = 'ARGM-TMP'
    sentences_present = [
        ('My brithday is tomorrow', 2, 3, label, 'be.02'),
        ('Her car arrives today', 2, 3, label, 'arrive.01'),
        ('The party starts this afternoon', 2, 4, label, 'start.01'),
        ('The concert begins tonight', 2, 3, label, 'begin.01'),
        ('The movie premiers this evening' , 2, 4, label, 'premier.01')
    ]
    t = Perturb.perturb(sentences_present, perturb_time_present, nsamples=50)
    data_tup = t.data
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-TMP-F - T2', capability='CA-TMP-F', description='X')
    return test

def test_3_CA_TMP_F():
    label = 'ARGM-TMP'
    sentences_future = [
        ('I will have a party tonight', 2, 5, label, 'have.04'),
        ('I will go to the beach next week', 2, 7, label, 'go.01'),
        ('They will visit their grandmother next month', 2, 6, label, 'visit.01'),
        ('She will have a party next year', 2, 6, label, 'have.04'),
        ('He will go to the park tomorrow', 2, 6, label, 'go.01'),
    ]

    t = Perturb.perturb(sentences_future, perturb_time_future, nsamples=25)
    data_tup = t.data

    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-TMP-F - T3', capability='CA-TMP-F', description='X')
    return test

def test_1_CA_DIR_P():
    label = 'ARGM-DIR'
    dataset_only_source = [
       ('The cat jumped from the roof', 2, 5, label, 'jump.01'),
       ('I biked from my house', 1, 4, label, 'bike.01'),
       ('The dog ran from the park', 2, 5, label, 'run.02'),
    ]

    data_tup = add_goal(dataset_only_source, nsamples=25)
    expect_fn = Expect.pairwise(change_pred)
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-DIR-P - T1', capability='CA-DIR-P', description='X', expect=expect_fn)
    return test

def test_2_CA_DIR_P():
    label = 'ARGM-DIR'
    dataset_only_goal = [
        ('Workers dumped the waste into a huge bin', 1, 6, label, 'dump.01'),
        ('No one wants to go home', 4, 5, label, 'go.01'),
        ('The cat jumped into the box', 2, 5, label, 'jump.01'),
        ('The dog ran into the park', 2, 5, label, 'run.02')
    ]

    data_tup = add_source(dataset_only_goal, nsamples=25)
    expect_fn = Expect.pairwise(change_pred)
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='CA-DIR-P - T2', capability='CA-DIR-P', description='X', expect=expect_fn)
    return test

def dict_fn(x):
    sent = x[0]
    return {'sent': sent, 'pred': sent.split(' ')[int(x[1])], 'arg_pos': int(x[2]), 'label': x[3], 'pred_frame': x[4]}

def load_tests():
    suite = TestSuite()
    suite.add(test_1_CA0_R())
    suite.add(test_1_CA0_D())
    suite.add(test_1_CA1_D())
    suite.add(test_1_CA_LOC_L())
    suite.add(test_1_CA_NEG_A())
    suite.add(test_1_CA_TMP_F()) 
    suite.add(test_2_CA_TMP_F()) 
    suite.add(test_3_CA_TMP_F()) 
    suite.add(test_1_CA_TMP_D())
    suite.add(test_1_CA_DIR_P())
    suite.add(test_2_CA_DIR_P())

    with open('test_data/raw_test_data.json', 'w') as f:
        json.dump(suite.to_dict(dict_fn), f)


if __name__ == '__main__':
    load_tests()