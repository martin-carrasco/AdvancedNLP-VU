import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from pp import pp_matrix


df = pd.read_csv('data/preds/ident.csv')

print(f'F1: {f1_score(df["y"], df["pred_y"], average="macro")}')
print(f'Precision: {precision_score(df["y"], df["pred_y"], average="macro")}')
print(f'Recall: {recall_score(df["y"], df["pred_y"], average="macro")}')

cm = confusion_matrix(df['y'], df['pred_y'])

print('Total tokens: ', cm.sum())

df_cm = pd.DataFrame(cm, index = ['Arg', '7Arg'], columns = ['Arg', '7Arg'])

sn.heatmap(df_cm, annot=True)
cmap = 'PuRd'
#pp_matrix(df_cm, cmap='rocket')

plt.xlabel('Predicted')
plt.ylabel('True')


plt.savefig('data/preds/ident.png')