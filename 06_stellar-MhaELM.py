#!/usr/bin/env python
# Created by "Thieu" at 11:38 AM, 14/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from data.data_loader import load_stellar
from intelelm import MhaElmClassifier
from utils.visualizer import draw_confusion_matrix
from pathlib import Path
import pandas as pd
import multiprocessing
import time
import config


dataset_name = "stellar"
PATH_SAVE = f"{config.PATH_SAVE}/{dataset_name}"
Path(PATH_SAVE).mkdir(parents=True, exist_ok=True)

## Load data object
data, scaler_X, scaler_y = load_stellar(f"data/{dataset_name}.csv", scaling_method="standard")
#### (100000, 18) x 3 labels (GALAXY, STAR, QSO)

def process_optimizer(idx, opt, data, list_paras):
    t_start = time.perf_counter()
    ## Create model
    model = MhaElmClassifier(layer_sizes=(12, 6), act_name="relu", obj_name="CEL",
                             optim=opt, optim_paras=list_paras[idx], verbose=False)
    # Train the model
    model.fit(X=data.X_train, y=data.y_train)
    # Test the model
    y_pred_train = model.predict(data.X_train)
    y_pred_test = model.predict(data.X_test)
    # Save metrics
    model.save_metrics(data.y_train, y_pred_train, list_metrics=config.LIST_METRICS,
                       save_path=PATH_SAVE, filename=f"{model.optimizer.name}-train-metrics.csv")
    model.save_metrics(data.y_test, y_pred_test, list_metrics=config.LIST_METRICS,
                       save_path=PATH_SAVE, filename=f"{model.optimizer.name}-test-metrics.csv")
    # Save loss train
    model.save_loss_train(save_path=PATH_SAVE, filename=f"{model.optimizer.name}-loss.csv")
    # Save confusion matrix
    draw_confusion_matrix(data.y_train, y_pred_train,
                          title=f"Confusion Matrix of {model.optimizer.name} on training set",
                          pathsave=f"{PATH_SAVE}/{model.optimizer.name}-train-cm.png")
    draw_confusion_matrix(data.y_test, y_pred_test, title=f"Confusion Matrix of {model.optimizer.name} on testing set",
                          pathsave=f"{PATH_SAVE}/{model.optimizer.name}-test-cm.png")
    # Save model
    model.save_model(save_path=PATH_SAVE, filename=f"{model.optimizer.name}-model.pkl")
    t_end = time.perf_counter() - t_start
    return list_paras[idx]["name"], t_end


if __name__ == "__main__":
    with multiprocessing.Pool(processes=len(config.LIST_OPTIMIZERS)) as pool:
        results = pool.starmap(process_optimizer,
                               [(idx, opt, data, config.LIST_PARAS) for idx, opt in enumerate(config.LIST_OPTIMIZERS)])

    t_dict = dict(results)
    df = pd.DataFrame.from_dict(t_dict, orient="index")
    df.to_csv(f"{PATH_SAVE}/time-mha-elm.csv")
