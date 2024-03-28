import pandas as pd
import torch
from ast import literal_eval
import pipelines
from datasets import Dataset
from utils import tokenize_and_align_labels
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


def start_tests(model_name):
    output_dict = {
        'capability': [],
        'test_name': [],
        'failure_rate': [],
        'exampe_cnt': []
    }

    model = AutoModelForTokenClassification.from_pretrained(f'martincc98/{model_name}')
    tokenizer = AutoTokenizer.from_pretrained(f'martincc98/{model_name}')

    df = pd.read_csv('test_data/challenge_ds.csv')
    df['label'] = df['label'].apply(lambda x: model.config.label2id[x])

    capabilities = df['capability'].unique()


    for capability in capabilities:
        print('Capaility:', capability)
        cap_df: pd.DataFrame = df[df['capability'] == capability]  
        tests = cap_df['test_name'].unique()
        for test in tests:
            print('Test:', test)
            cur_df = cap_df[cap_df['test_name'] == test]
            test_cnt = len(cur_df)
            succes_cnt = 0
            ds = Dataset.from_pandas(cur_df)

            inputs = ds.map(lambda x: tokenize_and_align_labels(tokenizer, x))
            inputs.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'sent'])
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
            best_classes = torch.argmax(logits, dim=2).tolist()
            for i, sent_labels in enumerate(best_classes):
                classes = sent_labels #[model.config.id2label[x] for x in sent_labels]
                true_classes = inputs['labels'][i]
                #print(inputs[i]['sent'])
                #print(classes)
                #print(true_classes.tolist())
                for j, true_class in enumerate(true_classes):
                    if true_class == -100:
                        continue
                    if classes[j] == true_class:
                        succes_cnt += 1
                #break
                #print(inputs['labels'][i])
            fr = 1 - (succes_cnt / test_cnt)
            output_dict['capability'].append(capability)
            output_dict['test_name'].append(test)
            output_dict['failure_rate'].append(fr)
            output_dict['exampe_cnt'].append(test_cnt)
            print('Failure rate: ', fr)
    df_metric = pd.DataFrame.from_dict(output_dict)
    df_metric.to_csv(f'test_data/metrics_{model_name}.csv')

if __name__ == '__main__':
    # model_name = 'srl_bert'
    model_name = 'srl_bert_advanced'
    start_tests(model_name)