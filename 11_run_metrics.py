#!/usr/bin/env python
# Created by "Thieu" at 08:26, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd


def get_metrics(data_name, path_read, path_save, model_names):

    train_dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{path_read}/{model_name}-train-metrics.csv")
        df["model"] = model_name
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)

    test_dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{path_read}/{model_name}-test-metrics.csv")
        df["model"] = model_name
        test_dfs.append(df)
    test_df = pd.concat(test_dfs, ignore_index=True)

    train_df.to_csv(f"{path_save}/{data_name}-train.csv", index=False)
    test_df.to_csv(f"{path_save}/{data_name}-test.csv", index=False)


model_names = [
    # "SVM", "KNN", "DT", "RF", "ABC", "GBC", "MLP", "ELM",
               "AGTO-ELM", "AVOA-ELM", "ARO-ELM", "HGSO-ELM", "EVO-ELM", "TLO-ELM"]

path = "history_new_02"

get_metrics(data_name="credit_score", path_read=f"{path}/credit_score", path_save=path, model_names=model_names)
get_metrics(data_name="digits", path_read=f"{path}/digits", path_save=path, model_names=model_names)
get_metrics(data_name="income", path_read=f"{path}/income", path_save=path, model_names=model_names)
get_metrics(data_name="loan_approval", path_read=f"{path}/loan_approval", path_save=path, model_names=model_names)
get_metrics(data_name="stroke_prediction", path_read=f"{path}/stroke_prediction", path_save=path, model_names=model_names)
get_metrics(data_name="stellar", path_read=f"{path}/stellar", path_save=path, model_names=model_names)
get_metrics(data_name="hotel_booking", path_read=f"{path}/hotel_booking", path_save=path, model_names=model_names)
get_metrics(data_name="mobile_price", path_read=f"{path}/mobile_price", path_save=path, model_names=model_names)
get_metrics(data_name="airline_passenger", path_read=f"{path}/airline_passenger", path_save=path, model_names=model_names)
get_metrics(data_name="bank_customer_churn", path_read=f"{path}/bank_customer_churn", path_save=path, model_names=model_names)
