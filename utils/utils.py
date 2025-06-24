from collections import defaultdict
import pandas as pd
import numpy as np

import torch
import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.itertools import batcher

def load_test_data(pred_length, context, quantiles, dataset, forecast_date):
    PDT = pred_length
    CTX = context

    # Load dataframe and GluonTS dataset
    df = pd.read_csv(dataset, index_col=0, parse_dates=['ds'])
    df['y'] = df['y'].astype(np.float32)
    # df = pd.read_csv(dataset, index_col=False, parse_dates=['ds'])
    ds = PandasDataset.from_long_dataframe(df, target="y", item_id="unique_id", timestamp='ds')
    freq = ds.freq
    unit = ''.join(char for char in freq if not char.isdigit())
    print(f'freq: {freq}, unit: {unit}')
    unit_str = "".join(filter(str.isdigit, freq))
    if unit_str == "":
        unit_num = 1
    else:
        unit_num = int("".join(unit_str))
    if unit == 'M':
        freq_delta = pd.DateOffset(months=unit_num)
    else:
        freq_delta = pd.Timedelta(unit_num, unit)

    
    if forecast_date == "":
        forecast_date = min(df['ds']) + freq_delta * CTX
    else:
        forecast_date = pd.Timestamp(forecast_date)
    end_date = max(df['ds'])
    if unit == 'M':
        total_forecast_length = (end_date.to_period(unit)-forecast_date.to_period(unit)).n // unit_num + 1
    else:
        total_forecast_length = (end_date-forecast_date) // freq_delta

    _, test_template = split(
        ds, date=pd.Period(forecast_date, freq=freq)
    )

    # Construct rolling window evaluation
    test_data = test_template.generate_instances(
        prediction_length=PDT,  # number of time steps for each prediction
        windows=total_forecast_length-PDT,  # number of windows in rolling window evaluation
        distance=1,  # number of time steps between each window - distance=PDT for non-overlapping windows
        max_history=CTX,
    )
    return test_data, freq, unit, freq_delta