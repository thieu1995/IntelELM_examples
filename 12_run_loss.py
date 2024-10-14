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
    merged_df.to_csv(f"history/{data_name}-loss.csv", index=False)

    # Plot the loss for all models in a single figure
    plt.figure(figsize=(8, 6))
    for model_name, group in merged_df.groupby('Model'):
        plt.plot(group['epoch'], group['loss'], label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Fitness value')
    plt.title("The fitness value of compared Metaheuristic-based ELM models")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"history/{data_name}-loss.png", bbox_inches='tight')
    if verbose:
        plt.show()


model_names = ["AGTO-ELM", "AVOA-ELM", "ARO-ELM", "HGSO-ELM", "EVO-ELM", "TLO-ELM"]

draw_loss(data_name="iris", pathfile="history/iris", model_names=model_names)
draw_loss(data_name="digits", pathfile="history/digits", model_names=model_names)
draw_loss(data_name="wine", pathfile="history/wine", model_names=model_names)
