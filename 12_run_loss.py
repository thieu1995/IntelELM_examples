#!/usr/bin/env python
# Created by "Thieu" at 08:41, 28/11/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import matplotlib.pyplot as plt


def draw_loss(data_name, pathfile, model_names, verbose=False):
    dfs = []
    for model_name in model_names:
        df = pd.read_csv(f"{pathfile}/{model_name}-loss.csv")
        df['Model'] = model_name
        dfs.append(df)
    # Concatenate all DataFrames in the list
    merged_df = pd.concat(dfs, ignore_index=True)
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(f"history_new/{data_name}-loss.csv", index=False)

    # Plot the loss for all models in a single figure
    plt.figure(figsize=(8, 6))
    for model_name, group in merged_df.groupby('Model'):
        plt.plot(group['epoch'], group['loss'], label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Fitness value')
    plt.title("The fitness value of compared Metaheuristic-based ELM models")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"history_new/{data_name}-loss.png", bbox_inches='tight')
    if verbose:
        plt.show()


model_names = ["AGTO-ELM", "AVOA-ELM", "ARO-ELM", "HGSO-ELM", "EVO-ELM", "TLO-ELM"]

draw_loss(data_name="credit_score", pathfile="history_new/credit_score", model_names=model_names)
draw_loss(data_name="digits", pathfile="history_new/digits", model_names=model_names)
draw_loss(data_name="income", pathfile="history_new/income", model_names=model_names)
draw_loss(data_name="loan_approval", pathfile="history_new/loan_approval", model_names=model_names)
draw_loss(data_name="stroke_prediction", pathfile="history_new/stroke_prediction", model_names=model_names)
draw_loss(data_name="stellar", pathfile="history_new/stellar", model_names=model_names)
draw_loss(data_name="hotel_booking", pathfile="history_new/hotel_booking", model_names=model_names)
draw_loss(data_name="mobile_price", pathfile="history_new/mobile_price", model_names=model_names)
draw_loss(data_name="airline_passenger", pathfile="history_new/airline_passenger", model_names=model_names)
draw_loss(data_name="bank_customer_churn", pathfile="history_new/bank_customer_churn", model_names=model_names)
