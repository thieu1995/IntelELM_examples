#!/bin/bash

# Chạy các script Python theo thứ tự, chỉ chạy tiếp nếu lệnh trước đó thành công
python 07_hotel_booking-MhaELM.py && \
python 07_hotel_booking-ML.py && \
python 08_mobile_price-MhaELM.py && \
python 08_mobile_price-ML.py && \
python 09_airline_passenger-MhaELM.py && \
python 09_airline_passenger-ML.py && \
python 10_bank_customer_churn-MhaELM.py && \
python 10_bank_customer_churn-ML.py && \

echo "All scripts finished successfully."
