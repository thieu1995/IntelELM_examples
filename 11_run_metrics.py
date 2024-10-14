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

    train_df.to_csv(f"history_new/{data_name}-train.csv", index=False)
    test_df.to_csv(f"history_new/{data_name}-test.csv", index=False)


model_names = ["SVM", "KNN", "DT", "RF", "ABC", "GBC", "MLP", "ELM",
               "AGTO-ELM", "AVOA-ELM", "ARO-ELM", "HGSO-ELM", "EVO-ELM", "TLO-ELM"]

get_metrics(data_name="credit_score", pathfile="history_new/credit_score", model_names=model_names)
get_metrics(data_name="digits", pathfile="history_new/digits", model_names=model_names)
get_metrics(data_name="income", pathfile="history_new/income", model_names=model_names)
get_metrics(data_name="loan_approval", pathfile="history_new/loan_approval", model_names=model_names)
get_metrics(data_name="stroke_prediction", pathfile="history_new/stroke_prediction", model_names=model_names)
get_metrics(data_name="stellar", pathfile="history_new/stellar", model_names=model_names)
get_metrics(data_name="hotel_booking", pathfile="history_new/hotel_booking", model_names=model_names)
get_metrics(data_name="mobile_price", pathfile="history_new/mobile_price", model_names=model_names)
get_metrics(data_name="airline_passenger", pathfile="history_new/airline_passenger", model_names=model_names)
get_metrics(data_name="bank_customer_churn", pathfile="history_new/bank_customer_churn", model_names=model_names)
