import random
import json
import itertools
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import pipeline
import numpy as np
import pdb

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


def test_1_PatientRight():
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

    test = MFT(data=data_tup, labels=arg_pos, name='PatientRight', capability='PatientRight', description='Patient(ARG1) is to the write of the predicate')

    return test

def test_1_PatientRole3D():
    nouns = ['person', 'child', 'animal', 'mailman']
    prop = ['aside', 'beside', 'by', 'near']
    add = ['that', 'this']
    nouns_two = ['car', 'house', 'road']
    adjs = ['green', 'blurry', 'sad', 'happy', 'angry']
    data = editor.template('The {nouns} {prop} {add} {nouns_two} is {adjs}',
                            prop=prop, nouns=nouns, adjs=adjs, nouns_two=nouns_two,
                            add=add,
                            save=True, labels=5, nsamples=25)

    pred_pos = 5
    arg_pos = 1
    label='ARG1'

    data_tup = [(x, pred_pos, arg_pos, label, 'be.02') for x in data.data]

    test = MFT(data=data_tup, labels=arg_pos, name='PatientRole3D', capability='PatientRole3D', description='Patient(ARG1) has a distance of 3 or more to the predicate')
    return test

def test_1_AgentRole3D():
    prep = ['Nobody', 'Anybody']
    en = ['that']
    adverb_a = ['unintentionally', 'accidentally', 'intentionally']
    ors = ['and', 'yet']
    adverb_b = ['happily', 'sadly', 'angrily', 'calmly']
    verbs = ['eats', 'drinks', 'cries', 'laughs']
    filler = ['in public', 'in private', 'in the bus']
    verbs_two = ['is']
    adj = ['trustworthy', 'unreliable', 'smart']
    data = editor.template('{prep} {en} {adverb_a} {ors} {adverb_b} {verbs} {filler} {verbs_two} {adj}',
                            ors=ors, adverb_a=adverb_a, adverb_b=adverb_b, en=en, verbs=verbs,
                            filler=filler, verbs_two=verbs_two, adj=adj,
                            prep=prep, nsamples=25)

    pred_pos = 5
    arg_pos = 0
    label = 'ARG0'
    verbs_frame =['eat.01', 'drink.01', 'watch.01', 'cry.01']

    data_tup = []
    for x in data.data:
        idx = verbs.index(x.split(' ')[5])
        data_tup.append((x, pred_pos, arg_pos, label, verbs_frame[idx]))

    test = MFT(data=data_tup, labels=arg_pos, name='AgentRole3D', capability='AgentRole3D', description='Agent(ARG0) has a distance of 3 or more to the predicate')
    return test
    
def test_1_LocationVar():
    label = 'ARGM-LOC'
    dataset = [
        ('I bought flowers in Miami', 1, 4, label, 'buy.01'),
        ('My mother used to live in Mexico', 4, 6, label, 'live.01'),
        ('There is a fantastic flower shop in China', 1, 7, label, 'be.02')
    ]
    t = Perturb.perturb(dataset, change_loc, nsamples=30, keep_original=False)
    data_tup = t.data

    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='LocationVar', capability='LocationVar', description='X')
    return test

def test_1_NegAdv():
    na = ['never']
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
    
    test = MFT(data=data_tup, labels=arg_pos, name='NegAdv', capability='NegAdv', description='X')
    return test

def test_1_TempAdv():
    label = 'ARGM-TMP'
    dataset = [
        ('My brithday will take place on Monday', 3, 6, label, 'take.01'),
        ('I visited my grandmother last May', 1, 5, label, 'visit.01'),
        ('I will have a party next Saturday', 2 , 6, label, 'have.04')
    ]

    t = Perturb.perturb(dataset, perturb_date_ref, nsamples=50)
    data_tup_raw = t.data
    data_tup = []
    for x in data_tup_raw:
        for y in x:
            data_tup.append((y[0], y[1], y[2], label, y[4]))

    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='TempAdv', capability='TempAdv', description='X')
    return test

def test_1_FreqAdvPast():
    label = 'ARGM-TMP'
    sentences_past = [
        ('I visited my grandmother recently', 1, 4, label, 'visit.01'),
        ('I ate a cake yesterday', 1, 4, label, 'eat.01'),
        ('He slept last night', 1, 3, label, 'sleep.01'),
        ('They went to the beach last week', 1, 6, label, 'go.01'),
        ('She cared for her dog last month', 1, 6, label, 'care.03')
    ]

    t = Perturb.perturb(sentences_past, perturb_time_past, nsamples=50)
    data_tup = t.data
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='FreqAdvPast', capability='FreqAdvPast', description='X')
    return test

def test_2_FreqAdvPresent():
    label = 'ARGM-TMP'
    sentences_present = [
        ('I will be having my birthday tomorrow', 3, 6, label, 'have.02'),
        ('Her car is arriving today', 3, 4, label, 'arrive.01'),
        ('The party is starting this afternoon', 3, 5, label, 'start.01'),
        ('The concert begins tonight', 2, 3, label, 'begin.01'),
        ('The movie premiers this evening' , 2, 4, label, 'premier.01')
    ]
    t = Perturb.perturb(sentences_present, perturb_time_present, nsamples=50)
    data_tup = t.data
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='FreqAdvPresent', capability='FreqAdvPresent', description='X')
    return test

def test_3_FreqAdvFuture():
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

    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='FreqAdvFuture', capability='FreqAdvFuture', description='X')
    return test

def test_1_VarDirTarget():
    label = 'ARGM-DIR'
    dataset_only_source = [
       ('The cat jumped forward from the roof', 2, 3, label, 'jump.01'),
       ('I biked from my house', 1, 4, label, 'bike.01'),
       ('The dog ran along the road', 2, 5, label, 'run.02'),
    ]

    data_tup = add_goal(dataset_only_source, nsamples=25)
    expect_fn = Expect.pairwise(change_pred)
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='VarDirTarget', capability='VarDirTarget', description='X', expect=expect_fn)
    return test

def test_2_VarGoalSource():
    label = 'ARGM-GOL'
    dataset_only_goal = [
        ('Workers dumped the waste from my house', 1, -1, label, 'dump.01'),
        ('My brother threw a paper', 2, -1, label, 'throw.01'),
        ('The laundry should be put', 4, -1, label, 'put.01'),
    ]

    data_tup = add_tar(dataset_only_goal, nsamples=25)
    expect_fn = Expect.pairwise(change_pred)
    test = INV(data=data_tup, labels=[x[2] for x in data_tup], name='VarGoalSource', capability='VarGoalSource', description='X', expect=expect_fn)
    return test

def add_tar(x, nsamples=10):
    phrases = ['into a huge bin', 'into the garbage disposal', 'into the red basket', 'onto the floor']
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
        data_tup = ('{} {}'.format(x_c, p), pred_pos, len(x_c.split(' ')) + len(p.split(' ')) - 1, label, pred_frame)
        rets.append(data_tup)
        cnt+=1

    return rets


def dict_fn(x):
    #pdb.set_trace()
    sent = x[0]
    return {'sent': sent, 'pred': sent.split(' ')[int(x[1])], 'pred_pos': int(x[1]), 'arg_pos': int(x[2]), 'label': x[3], 'pred_frame': x[4]}

def load_tests():
    suite = TestSuite()
    suite.add(test_1_PatientRight())
    suite.add(test_1_AgentRole3D())
    suite.add(test_1_PatientRole3D())
    suite.add(test_1_TempAdv())
    suite.add(test_1_NegAdv())
    suite.add(test_1_LocationVar())
    suite.add(test_1_FreqAdvPast())
    suite.add(test_2_FreqAdvPresent())
    suite.add(test_3_FreqAdvFuture())
    suite.add(test_1_VarDirTarget())
    suite.add(test_2_VarGoalSource())

    base_dict = suite.to_dict(dict_fn)
    # (i) -> {'sent': sent, ... }
    final_dict = {}

    # Get the categories of the tests
    test_names = set(base_dict['test_name'])
    test_num = len(list(base_dict.values())[0])

    for test_name in test_names:
        test_name_list = []
        for i in range(test_num):
            if base_dict['test_name'][i] != test_name:
                continue
            current_dict = {}
            for k in list(base_dict.keys()):
                current_dict[k] = base_dict[k][i]
            test_name_list.append(current_dict)
        final_dict[test_name] = test_name_list

    with open('test_data/human_readable_test_data.json', 'w') as f:
        json.dump(final_dict, f)
    with open('test_data/raw_test_data.json', 'w') as f:
        json.dump(base_dict, f)
    



if __name__ == '__main__':
    load_tests()