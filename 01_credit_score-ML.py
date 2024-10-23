#!/usr/bin/env python
# Created by "Thieu" at 4:04 PM, 14/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from data.data_loader import load_credit_score
from intelelm import ElmClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from utils.helper import save_metrics, save_model
from utils.visualizer import draw_confusion_matrix
from pathlib import Path
import pandas as pd
import time
import config


TEST_SIZE = 0.2
dataset_name = "credit_score"
PATH_SAVE = f"{config.PATH_SAVE}/{dataset_name}"
Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

## Load data object
data, scaler_X, scaler_y = load_credit_score(f"data/{dataset_name}.csv", scaling_method="standard")
## 100000 x 28 x 3 labels

classifiers = {
    "SVM": SVC(kernel="rbf", C=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=25),
    "DT": DecisionTreeClassifier(max_depth=6, random_state=42),
    "RF": RandomForestClassifier(max_depth=6, n_estimators=100, max_features=5, random_state=42),
    "ABC": AdaBoostClassifier(n_estimators=100, learning_rate=0.4, random_state=42),
    "GBC": GradientBoostingClassifier(n_estimators=100, learning_rate=0.4, max_depth=5, random_state=42),
    "MLP": MLPClassifier(alpha=1, max_iter=1000, hidden_layer_sizes=(40, 15), activation="relu", random_state=42),
    "ELM": ElmClassifier(layer_sizes=(60, 30), act_name="elu")
}

t_dict = {}
for name, model in classifiers.items():
    t_start = time.perf_counter()

    ## Train the model
    model.fit(X=data.X_train, y=data.y_train)

    ## Test the model
    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)

    ## Save metrics
    save_metrics(problem="classification", y_true=data.y_train, y_pred=y_pred_train,
                 list_metrics=config.LIST_METRICS,
                 save_path=PATH_SAVE, filename=f"{name}-train-metrics.csv")
    save_metrics(problem="classification", y_true=data.y_test, y_pred=y_pred_test,
                 list_metrics=config.LIST_METRICS,
                 save_path=PATH_SAVE, filename=f"{name}-test-metrics.csv")

    ## Save confusion matrix
    draw_confusion_matrix(data.y_train, y_pred_train, title=f"Confusion Matrix of {name} on training set",
                          pathsave=f"{PATH_SAVE}/{name}-train-cm.png")
    draw_confusion_matrix(data.y_test, y_pred_test, title=f"Confusion Matrix of {name} on testing set",
                          pathsave=f"{PATH_SAVE}/{name}-test-cm.png")
    ## Save model
    save_model(model=model, save_path=PATH_SAVE, filename=f"{name}-model.pkl")

    t_end = time.perf_counter() - t_start
    t_dict[name] = t_end

df = pd.DataFrame.from_dict(t_dict, orient="index")
df.to_csv(f"{PATH_SAVE}/time-ml.csv")
