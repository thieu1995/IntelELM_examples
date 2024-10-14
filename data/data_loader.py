#!/usr/bin/env python
# Created by "Thieu" at 10:32 AM, 14/10/2024 -------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from intelelm import Data
import pandas as pd
from imblearn.over_sampling import SMOTE


def load_income(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/lodetomasi1995/income-classification/data
    # https://www.kaggle.com/code/avantikab/income-classification-eda-and-6-ml-models
    # https://www.kaggle.com/code/thieunv/income-eda-classification-intelelm-80/edit
    df = pd.read_csv(path_file)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="Income")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_credit_score(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data
    # https://www.kaggle.com/code/metehanbayraktar/credit-score-classification-with-ann-92-87#ANN-Model
    # https://www.kaggle.com/code/thieunv/credit-score-classification-intelelm-90/edit
    df = pd.read_csv(path_file)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="CreditScore")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_mobile_price(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data
    # https://www.kaggle.com/code/gulsahdemiryurek/mobile-price-classification-with-svm/notebook
    # https://www.kaggle.com/code/thieunv/glass-classification-intelelm-98/edit
    df = pd.read_csv(path_file)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="MobilePrice")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_glass_classification(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/uciml/glass/data
    # https://www.kaggle.com/code/lucifer959999/glass-classification-binary-accuracy-of-97
    # https://www.kaggle.com/code/thieunv/glass-classification-intelelm-98/edit
    df = pd.read_csv(path_file)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="GlassClassification")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_bank_customer_churn(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data
    # https://www.kaggle.com/code/leo21892/bank-customer-churn-prediction
    # https://www.kaggle.com/code/thieunv/bank-customer-churn-classification-intelelm/edit
    df = pd.read_csv(path_file)

    # Convert boolean column 'A' to integer
    df['country_Germany'] = df['country_Germany'].astype(int)
    df['country_Spain'] = df['country_Spain'].astype(int)
    df['gender_Male'] = df['gender_Male'].astype(int)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="BankCustomerChurn")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_airline_passenger(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
    # https://www.kaggle.com/code/manarmohamed24/airline-passenger-satisfaction-acc-95-7
    # https://www.kaggle.com/code/thieunv/airline-passenger-satisfaction-intelelm/edit
    df = pd.read_csv(path_file, header=0, index_col=False, skiprows=1)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="AirlinePassenger")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_email_spam(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv/data
    # https://www.kaggle.com/code/shiva51g/email-spam-classifier
    # https://www.kaggle.com/code/thieunv/email-spam-classification-intelelm/edit
    df = pd.read_csv(path_file, header=0, index_col=False, skiprows=1)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="EmailSpam")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_hotel_booking(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/mojtaba142/hotel-booking/data
    # https://www.kaggle.com/code/thieunv/hotel-booking-eda-classification-intelelm/edit
    df = pd.read_csv(path_file, header=0, index_col=False, skiprows=1)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="HotelBooking")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_stellar(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17/data
    # https://www.kaggle.com/code/thieunv/stellar-classification-intelelm
    df = pd.read_csv(path_file, header=0, index_col=False, skiprows=1)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="Stellar")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


def load_stroke_prediction(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data
    # https://www.kaggle.com/code/thieunv/stroke-prediction-intelelm/edit
    df = pd.read_csv(path_file, header=0, index_col=False, skiprows=1)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    smote = SMOTE()
    x_smote, y_smote = smote.fit_resample(X, y)

    data = Data(x_smote, y_smote, name="StrokePrediction")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y_smote)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method,))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None



def load_loan_approval(path_file, test_size=0.2, seed=2, shuffle=True, scaling_method="standard"):
    # https://www.kaggle.com/datasets/itshappy/ps4e9-original-data-loan-approval-prediction/data
    # https://www.kaggle.com/code/thieunv/loan-approval-prediction-intelelm/edit
    # https://www.kaggle.com/code/karnavivek/loan-approval-prediction-using-logistic-regression

    df = pd.read_csv(path_file, header=None, index_col=False, skiprows=1)

    # Split data into Features and Target
    X = df.values[:, :-1]
    y = df.values[:, -1:]

    data = Data(X, y, name="LoanApproval")

    ## Split train and test
    data.split_train_test(test_size=test_size, random_state=seed, inplace=True, shuffle=shuffle, stratify=y)

    ## Scaling dataset
    data.X_train, scaler_X = data.scale(data.X_train, scaling_methods=(scaling_method, ))
    data.X_test = scaler_X.transform(data.X_test)

    # data.y_train, scaler_y = data.encode_label(data.y_train)
    # data.y_test = scaler_y.transform(data.y_test)

    return data, scaler_X, None


