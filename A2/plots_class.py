import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from pp import pp_matrix


df = pd.read_csv('data/preds/classf.csv')

print(f'F1: {f1_score(df["y"], df["pred_y"], average="macro")}')
print(f'Precision: {precision_score(df["y"], df["pred_y"], average="macro")}')
print(f'Recall: {recall_score(df["y"], df["pred_y"], average="macro")}')

cm = confusion_matrix(df['y'], df['pred_y'])

columns = df['y'].unique()



df_cm = pd.DataFrame(cm, index = columns, columns = columns)

plt.figure(figsize=(14, 14))
sn.heatmap(df_cm, annot=True, annot_kws={'size': 5})
plt.xlabel('Predicted')
plt.ylabel('True')

#cmap = 'PuRd'
#pp_matrix(df_cm, cmap=cmap)


plt.savefig('data/preds/classf.png')