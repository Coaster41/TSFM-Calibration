from collections import defaultdict
import pandas as pd
import numpy as np
from chronos import ChronosBoltPipeline
import multiprocessing

import torch
import argparse
import os
import time
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.itertools import batcher
from utils.utils import load_test_data

PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer
VOCAB_SIZE = 4096

def load_model():
    return ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )


def run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, save_dir, pipeline=None):    
    if pipeline == None:
        pipeline = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
    pipeline.model.config.top_k = VOCAB_SIZE

    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    id = -1
    for batch in batcher(test_data.input, batch_size=BSZ):
        context = [torch.tensor(entry["target"]) for entry in batch]
        quantile_forecasts, mean_forecasts = pipeline.predict_quantiles(
            context=context,
            prediction_length=pred_length,
            quantile_levels=[0.5, *(np.array(quantiles)/100)],
        )
        mean_forecasts = mean_forecasts.detach().cpu().numpy()
        quantile_forecasts = quantile_forecasts.detach().cpu().numpy()
        for entry, quantile_forecast, mean_forecast in zip(batch, quantile_forecasts, mean_forecasts):
            if id != entry["item_id"]:
                id = entry["item_id"]
                print(f"Run Time: {time.time()-start_time:.2f}, ID: {id}")
            start_date = entry["start"] + (len(entry["target"])-1) 
            mean_results.append([id, start_date, *mean_forecast])
            median_results.append([id, start_date, *quantile_forecast[:,0]])
            for i in range(len(quantiles)):
                quantile_results[i].append([id, start_date, *quantile_forecast[:,i+1]])

    print(f'done: {time.time()-start_time:.2f}')

    os.makedirs(save_dir, exist_ok=True)
    columns = ['unique_id', 'ds', *range(1,pred_length+1)]
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
    run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, args.save_dir)