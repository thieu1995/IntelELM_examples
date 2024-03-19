#!/usr/bin/env python
# Created by "Thieu" at 02:07, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def draw_confusion_matrix(y_true, y_pred, figsize=(8, 6), title='Confusion Matrix',
                          pathsave="history/cm.png", verbose=False):
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)

    # Save the confusion matrix as an image (optional)
    plt.savefig(pathsave, bbox_inches='tight')

    if verbose:
        plt.show()
