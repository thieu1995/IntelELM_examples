#!/bin/bash

# Chạy các script Python theo thứ tự, chỉ chạy tiếp nếu lệnh trước đó thành công
python 01_credit_score-MhaELM.py && \
python 02_digits-MhaELM.py && \
python 03_income-MhaELM.py && \
python 04_loan_approval-MhaELM.py && \
python 05_stroke_prediction-MhaELM.py && \
python 06_stellar-MhaELM.py && \
python 07_hotel_booking-MhaELM.py && \
python 08_mobile_price-MhaELM.py && \
python 09_airline_passenger-MhaELM.py && \
python 10_bank_customer_churn-MhaELM.py && \

echo "All scripts finished successfully."
