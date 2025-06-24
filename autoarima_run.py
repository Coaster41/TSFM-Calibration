from collections import defaultdict
import pandas as pd
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast.arima import arima_string

import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset

PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer


def run_model(dataset_fn, quantiles, pred_length, unit, freq, freq_delta, save_dir, context_length, forecast_date, season):
    os.makedirs(save_dir, exist_ok=True)

    # Load dataframe and GluonTS dataset
    df = pd.read_csv(dataset_fn, index_col=0, parse_dates=['ds'])    
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
        forecast_date = min(df['ds']) + context_length * freq_delta
    else:
        forecast_date = pd.Timestamp(forecast_date)
    end_date = max(df['ds'])

    train_df = df.loc[df['ds'] <= forecast_date]

    # Load Model
    model = AutoARIMA(season_length=season)
    
    sf = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=-1
    )
    
    start_time = time.time()
    sf.fit(df=train_df)
    arima_params = arima_string(sf.fitted_[0,0].model_)
    open_1 = arima_params.find("(")
    close_1 = arima_params.find(")")
    (p, q, d) = [int(i) for i in arima_params[open_1+1:close_1].split(",")]
    if arima_params.find("(", open_1+1) == -1:
        P, D, Q = 0, 0, 0
    else:
        (P, D, Q) = [int(i) for i in arima_params[arima_params.find("(", open_1+1)+1 \
                                              :arima_params.find(")", close_1+1)].split(",")]
    model = ARIMA(order=(p,q,d), season_length=season, seasonal_order=(P,D,Q))
    sf = StatsForecast(
        models=[model],
        freq=freq,
        n_jobs=-1
    )
    print(f'Finshed fitting in {time.time()-start_time:.2f}')
    
    forecast_cols = ["ARIMA", "ARIMA-lo-0.5",  \
                        *[f"ARIMA-lo-{quantile}" for quantile in quantiles[len(quantiles)//2:]], \
                        *[f"ARIMA-hi-{quantile}" for quantile in quantiles[len(quantiles)//2:]]]
    file_names = ["mean_preds", "median_preds", \
                        *[f"quantile_{100-quantile}_preds" for quantile in quantiles[len(quantiles)//2:]], \
                        *[f"quantile_{quantile}_preds" for quantile in quantiles[len(quantiles)//2:]]] 
    model_results = defaultdict(list)
    date_range = forecast_date + np.arange((end_date.to_period(freq) - forecast_date.to_period(freq)).n//unit_num + 1) * freq_delta
    for last_observed in date_range:
        forecast_df = sf.forecast(df=(df.loc[df['ds']<=last_observed]), h=pred_length, level=[0.5, *quantiles[len(quantiles)//2:]])
        print(f"Time: {time.time()-start_time:.4f}\t{last_observed}")
        # print(arima_string(sf.fitted_[0,0].model_))
        for forecast_col, file_name in zip(forecast_cols, file_names):
            forecast_result = pd.DataFrame(forecast_df[['unique_id', forecast_col]].groupby('unique_id')[forecast_col].agg(list), 
                                            columns=[forecast_col])
            forecast_result[list(range(1,pred_length+1))] = pd.DataFrame(forecast_result[forecast_col].tolist(), 
                                                                index=forecast_result.index)
            forecast_result.drop(columns=[forecast_col], inplace=True)
            forecast_result.insert(0, 'ds', last_observed)
            forecast_result = forecast_result.reset_index(drop=False)
            model_results[file_name].append(forecast_result)
    
    for file_name in file_names:
        forecast_result = pd.concat(model_results[file_name], ignore_index=True)
        forecast_result.to_csv(f"{save_dir}/{file_name}.csv")

    print(f'Done in {time.time()-start_time:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a model and dataset, then make predictions."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to dataset"
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Path to save results"
    )
    parser.add_argument(
        "--season", type=int, required=True, help="season length"
    )
    parser.add_argument(
        "--context", type=int, default=512, help="Size of context"
    )
    parser.add_argument(
        "--pred_length", type=int, default=24, help="Prediction horizon length"
    )
    parser.add_argument(
        "--quantiles", type=str, default="10,90", help="Prediction quantiles (comma delimited)"
    )
    parser.add_argument(
        "--forecast_date", type=str, default="", help="Date to start forecasting from"
    )

    args = parser.parse_args()
    PDT = args.pred_length
    CTX = args.context
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]
    run_model(args.dataset, quantiles, PDT, None, None, None, args.save_dir, CTX, args.forecast_date, args.season)

    