#!/usr/bin/env python
# Created by "Thieu" at 03:02, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from sklearn.datasets import load_wine
from intelelm import Data, ElmClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from utils.helper import save_metrics, save_model
from utils.visualizer import draw_confusion_matrix
from pathlib import Path

# 13 inputs, 3 outputs, 178 samples

TEST_SIZE = 0.2
dataset_name = "wine"
PATH_SAVE = f"history/{dataset_name}"
Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

## Load data object
X, y = load_wine(return_X_y=True)
data = Data(X, y, name=dataset_name)

## Split train and test
data.split_train_test(test_size=TEST_SIZE, random_state=2, inplace=True, shuffle=True)

## Scaling dataset
data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=("standard"))
data.X_test = scaler_X.transform(data.X_test)

data.y_train, scaler_y = data.encode_label(data.y_train)
data.y_test = scaler_y.transform(data.y_test)

classifiers = {
    "SVM": SVC(kernel="linear", C=0.1, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=3),
    "DT": DecisionTreeClassifier(max_depth=4, random_state=42),
    "RF": RandomForestClassifier(max_depth=4, n_estimators=25, max_features=2, random_state=42),
    "ABC": AdaBoostClassifier(n_estimators=25, learning_rate=0.5, random_state=42),
    "GBC": GradientBoostingClassifier(n_estimators=40, learning_rate=0.5, max_depth=3, random_state=42),
    "MLP": MLPClassifier(alpha=1, max_iter=750, hidden_layer_sizes=(10,), activation="relu", random_state=42),
    "ELM": ElmClassifier(hidden_size=20, act_name="elu")
}

for name, model in classifiers.items():
    ## Train the model
    model.fit(X=data.X_train, y=data.y_train)

    ## Test the model
    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)

    ## Save metrics
    save_metrics(problem="classification", y_true=data.y_train, y_pred=y_pred_train,
                 list_metrics=("AS", "PS", "NPV", "RS", "F1S"),
                 save_path=PATH_SAVE, filename=f"{name}-train-metrics.csv")
    save_metrics(problem="classification", y_true=data.y_test, y_pred=y_pred_test,
                 list_metrics=("AS", "PS", "NPV", "RS", "F1S"),
                 save_path=PATH_SAVE, filename=f"{name}-test-metrics.csv")

    ## Save confusion matrix
    draw_confusion_matrix(data.y_train, y_pred_train, title=f"Confusion Matrix of {name} on training set",
                          pathsave=f"{PATH_SAVE}/{name}-train-cm.png")
    draw_confusion_matrix(data.y_test, y_pred_test, title=f"Confusion Matrix of {name} on testing set",
                          pathsave=f"{PATH_SAVE}/{name}-test-cm.png")
    ## Save model
    save_model(model=model, save_path=PATH_SAVE, filename=f"{name}-model.pkl")
