from transformers import Pipeline
from utils import tokenize_and_align_labels
import datasets
import torch


# TODO
''' Further investigate why using pipeline does not work. It pulls the model from the
hub but using random weights instead of the already trained weights. Also, it does not 
output the results according to the padding that is applied to the inputs
'''
class SRLPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        postprocess_kwargs = {}
        preprocess_kwargs = {}
        if 'model_type' in kwargs:
            preprocess_kwargs['model_type'] = kwargs['model_type']
        return preprocess_kwargs, {}, postprocess_kwargs

    def map_to_id(self, row):
        if type(row['label']) == list:
            int_rep = [self.model.config.label2id[x] for x in row['label']]
        else:
            int_rep = self.model.config.label2id[row['label']]
        return {'label': int_rep}

    def preprocess(self, ds: datasets.Dataset, model_type=None):
        inputs = ds.map(self.map_to_id)
        inputs = inputs.map(lambda x: tokenize_and_align_labels(self.tokenizer, x))
        inputs.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        best_classes = torch.argmax(model_outputs["logits"], dim=1).tolist()
        best_classes = [self.model.config.id2label[x] for x in best_classes]
        return {'labels': best_classes}

from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification

PIPELINE_REGISTRY.register_pipeline(
    "srl",
    pipeline_class=SRLPipeline,
    pt_model=AutoModelForSequenceClassification,
)