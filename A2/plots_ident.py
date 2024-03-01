import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from pp import pp_matrix


df = pd.read_csv('data/preds/ident.csv')

labels = list(set(df['y'].unique()))
df_dict = {
    'label': [],
    'f1': [],
    'precision': [],
    'recall': []
}

f1_scores =  f1_score(df["y"], df["pred_y"], average=None, labels=labels, zero_division=0)
recall_scores =  recall_score(df["y"], df["pred_y"], average=None, labels=labels, zero_division=0)
precision_scores =  precision_score(df["y"], df["pred_y"], average=None, labels=labels, zero_division=0)

for label, f1_s, recall_s, precision_s in zip(labels, f1_scores, recall_scores, precision_scores):
    df_dict['label'].append(label)
    df_dict['f1'].append(f1_s)
    df_dict['precision'].append(precision_s)
    df_dict['recall'].append(recall_s)

df_dict['label'].append('macro')
df_dict['f1'].append(f1_score(df["y"], df["pred_y"], average="macro", zero_division=0))
df_dict['precision'].append(precision_score(df["y"], df["pred_y"], average="macro", zero_division=0))
df_dict['recall'].append(recall_score(df["y"], df["pred_y"], average="macro", zero_division=0))

df_dict['label'].append('weighted')
df_dict['f1'].append(f1_score(df["y"], df["pred_y"], average="weighted", zero_division=0))
df_dict['precision'].append(precision_score(df["y"], df["pred_y"], average="weighted", zero_division=0))
df_dict['recall'].append(recall_score(df["y"], df["pred_y"], average="weighted", zero_division=0))

df_scores = pd.DataFrame.from_dict(df_dict)
df_scores.to_csv('data/preds/ident_scores.csv', index=False)

cm = confusion_matrix(df['y'], df['pred_y'])

print('Total tokens: ', cm.sum())

df_cm = pd.DataFrame(cm, index = ['Arg', '7Arg'], columns = ['Arg', '7Arg'])

sn.heatmap(df_cm, annot=True)
cmap = 'PuRd'
#pp_matrix(df_cm, cmap='rocket')

plt.xlabel('Predicted')
plt.ylabel('True')


plt.savefig('data/preds/ident.png')