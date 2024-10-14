#!/usr/bin/env python
# Created by "Thieu" at 21:56, 20/03/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
import matplotlib.pyplot as plt


def draw_computation_time(data_name, pathfile, verbose=False):
    # Setting a default color cycle for distinct colors
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    colors = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f'  # Gray
    ]

    # Read the CSV file into a DataFrame
    df1 = pd.read_csv(f"{pathfile}/{data_name}/time-mha-elm.csv")

    # Create a bar chart
    plt.figure(figsize=(6, 4))  # Adjust the figure size as needed
    plt.bar(df1['Model'], df1['Time'], color=colors, width=0.5)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Computation time (seconds)')
    plt.title('The computation time of Metaheuristic-based ELM models.')
    # Rotate x-axis labels for better readability (if needed)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"{pathfile}/{data_name}-time-mha-elm.png", bbox_inches='tight')
    if verbose:
        plt.show()

    # Read the CSV file into a DataFrame
    df2 = pd.read_csv(f"{pathfile}/{data_name}/time-ml.csv")
    # Create a bar chart
    plt.figure(figsize=(6, 4))  # Adjust the figure size as needed
    plt.bar(df2['Model'], df2['Time'], color=colors, width=0.5)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Computation time (seconds)')
    plt.title('The computation time of ML models.')
    # Rotate x-axis labels for better readability (if needed)
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"{pathfile}/{data_name}-time-ml.png", bbox_inches='tight')
    if verbose:
        plt.show()


draw_computation_time(data_name="credit_score", pathfile="history_new")
draw_computation_time(data_name="digits", pathfile="history_new")
draw_computation_time(data_name="income", pathfile="history_new")
draw_computation_time(data_name="loan_approval", pathfile="history_new")
draw_computation_time(data_name="stroke_prediction", pathfile="history_new")
draw_computation_time(data_name="stellar", pathfile="history_new")
draw_computation_time(data_name="hotel_booking", pathfile="history_new")
draw_computation_time(data_name="mobile_price", pathfile="history_new")
draw_computation_time(data_name="airline_passenger", pathfile="history_new")
draw_computation_time(data_name="bank_customer_churn", pathfile="history_new")
