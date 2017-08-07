import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, title_text='Confusion Matrix', file_name="Confusion_Matrix.png"):
    plt.figure(figsize=(25, 8))
    cm = confusion_matrix(y_true, y_pred)
    acc = np.sum(cm.diagonal()) / np.sum(cm)
    cm_2 = cm / np.sum(cm, axis=1)[:, np.newaxis]
    # cm_2 = np.append(cm_2, np.sum(cm, axis=1).reshape(len(labels), 1), axis=1)
    # x_labels = np.append(labels, 'Support')
    sns.heatmap(cm_2, annot=True, fmt='.2f', vmin=0, vmax=1)
    plt.title(title_text, fontsize=20)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual Class', fontsize=15)
    plt.yticks(rotation=0)
    plt.savefig(file_name)