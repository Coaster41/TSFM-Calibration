import pandas as pd
import numpy as np
import argparse
import os
import time

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from utils.utils import load_test_data

MODEL = "moirai"  # model name: choose from {'moirai', 'moirai-moe'}
BSZ = 32  # batch size: any positive integer


def load_model(PDT, CTX):
    return MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-small"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=32,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
    )

def run_model(test_data, quantiles, PDT, unit, freq, freq_delta, save_dir, CTX, model=None):

    if model == None:
        model = load_model(PDT, CTX)

    predictor = model.create_predictor(batch_size=BSZ)
    forecasts = predictor.predict(test_data.input)
    
    forecast_it = iter(forecasts)

    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    for i, forecast in enumerate(forecast_it):
        start_date = forecast.index[0] - 1
        mean_results.append([forecast.item_id, start_date, *np.mean(forecast.samples, axis=0)])
        median_results.append([forecast.item_id, start_date, *np.median(forecast.samples, axis=0)])
        for i, quantile in enumerate(quantiles):
            quantile_results[i].append([forecast.item_id, start_date, \
                                        *np.quantile(forecast.samples, q=quantile/100, axis=0)])
    print(f"done: {time.time()-start_time}")
    os.makedirs(save_dir, exist_ok=True)

    columns = ['unique_id', 'ds', *range(1,PDT+1)]
    mean_results = pd.DataFrame(mean_results, columns=columns)
    mean_results.to_csv(f"{save_dir}/mean_preds.csv")
    median_results = pd.DataFrame(median_results, columns=columns)
    median_results.to_csv(f"{save_dir}/median_preds.csv")
    for i, quantile in enumerate(quantiles):
        quantile_result = pd.DataFrame(quantile_results[i], columns=columns)
        quantile_results[i] = quantile_result
        quantile_result.to_csv(f"{save_dir}/quantile_{quantile}_preds.csv")




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
    pred_length = args.pred_length
    context = args.context
    dataset = args.dataset
    forecast_date = args.forecast_date
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]

    test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, dataset, forecast_date)
    run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, args.save_dir, context)