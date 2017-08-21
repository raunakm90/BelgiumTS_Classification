import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, title_text='Confusion Matrix', file_name="Confusion_Matrix.png"):
    plt.figure(figsize=(25, 10))
    sns.set(font_scale=1.5)
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat.shape)
    acc = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)
    conf_mat = conf_mat / np.sum(conf_mat, axis=1)[:, np.newaxis]
    mask = conf_mat == 0
    with sns.axes_style('white'):
        sns.heatmap(conf_mat, fmt='.2f', vmin=0, vmax=1,
                    mask=mask)
    plt.title(title_text + "\nAccuracy: %0.2f" % (acc * 100), fontsize=20)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('Actual Class', fontsize=15)
    plt.yticks(rotation=0)
    plt.savefig(file_name)
