#!/usr/bin/env python
# Created by "Thieu" at 08:26, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd


def get_metrics(data_name, pathfile, model_names):

    train_dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{pathfile}/{model_name}-train-metrics.csv")
        df["model"] = model_name
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)

    test_dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{pathfile}/{model_name}-test-metrics.csv")
        df["model"] = model_name
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df.to_csv(f"history/{data_name}-train.csv", index=False)
    test_df.to_csv(f"history/{data_name}-test.csv", index=False)


model_names = ["SVM", "KNN", "DT", "RF", "ABC", "GBC", "MLP", "ELM",
               "AGTO-ELM", "AVOA-ELM", "ARO-ELM", "HGSO-ELM", "EVO-ELM", "TLO-ELM"]

get_metrics(data_name="iris", pathfile="history/iris", model_names=model_names)
get_metrics(data_name="digits", pathfile="history/digits", model_names=model_names)
get_metrics(data_name="wine", pathfile="history/wine", model_names=model_names)
