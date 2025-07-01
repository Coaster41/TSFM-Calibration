import pandas as pd
import timesfm

import torch
import argparse
import os
import time
from gluonts.itertools import batcher
from utils.utils import load_test_data

BSZ = 32  # batch size: any positive integer

def load_model(PDT,CTX):
    return timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend='gpu',
            # per_core_batch_size=32,
            context_len=CTX,  # currently max supported
            horizon_len=PDT,  # number of steps to predict
            input_patch_len=32,  # fixed parameters
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
            point_forecast_mode='mean'
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
    )

def run_model(test_data, quantiles, PDT, unit, freq, freq_delta, save_dir, CTX, tfm=None):
    freq_id = {"M":1, "W":1, "D":0, "h":0, "min":0, "s":0, 'T':0, 'S':0, 'H':0}[unit]

    # Load Model
    if tfm == None:
        tfm = load_model(PDT,CTX)
    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    id = -1
    for batch in batcher(test_data.input, batch_size=BSZ):
        context = [torch.tensor(entry["target"]) for entry in batch]
        _, quantile_forecasts = tfm.forecast(context,
                                            freq=[freq_id] * len(context))
        for entry, forecasts in zip(batch, quantile_forecasts):
            if id != entry["item_id"]:
                id = entry["item_id"]
                print(f"Run Time: {time.time()-start_time:.2f}, ID: {id}")
            start_date = entry["start"] + (len(entry["target"])-1)
            mean_results.append([id, start_date, *forecasts[:,0]])
            median_results.append([id, start_date, *forecasts[:,5]])
            for i in range(len(quantiles)):
                quantile_results[i].append([id, start_date, *forecasts[:,quantiles[i]//10]])

    print(f'done: {time.time()-start_time:.2f}')

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