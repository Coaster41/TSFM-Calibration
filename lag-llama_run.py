import pandas as pd
import numpy as np

import torch
from lag_llama.gluon.estimator import LagLlamaEstimator

import torch
import argparse
import os
import time
from utils.utils import load_test_data

PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 100  # test set length: any positive integer
num_samples = 100


def load_model(PDT, CTX, ckpt_path="model_ckpts/lag-llama.ckpt"):
    gpu_num = 0
    device = torch.device(f"cuda:{gpu_num}") if torch.cuda.is_available() else torch.device('cpu')
    # Load Model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    return LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=PDT,
        context_length=CTX, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=None, 

        batch_size=BSZ,
        num_parallel_samples=100,
        device=device,
    )


def run_model(test_data, quantiles, PDT, unit, freq, freq_delta, save_dir, ckpt_path, CTX, estimator=None): 
    if estimator == None:
        estimator = load_model(PDT, CTX, ckpt_path)

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    forecasts = predictor.predict(test_data.input)
    forecast_it = iter(forecasts)

    mean_results = []
    median_results = []
    quantile_results = [[] for _ in quantiles]
    start_time = time.time()
    for i, (forecast) in enumerate(forecast_it):
        start_date = forecast.index[0] - 1
        print(f"time: {time.time()-start_time:.2f} date: {start_date} id: {forecast.item_id}")
        mean_results.append([forecast.item_id, start_date, *np.mean(forecast.samples, axis=0)])
        median_results.append([forecast.item_id, start_date, *np.median(forecast.samples, axis=0)])
        for i, quantile in enumerate(quantiles):
            quantile_results[i].append([forecast.item_id, start_date, \
                                        *np.quantile(forecast.samples, q=quantile/100, axis=0)])

    print(f'Done in {time.time()-start_time:.2f}')

    os.makedirs(args.save_dir, exist_ok=True)
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
    parser.add_argument(
        "--ckpt_path", type=str, default="model_ckpts/lag-llama.ckpt", help="Path to model checkpoint"
    )

    args = parser.parse_args()
    pred_length = args.pred_length
    context = args.context
    dataset = args.dataset
    forecast_date = args.forecast_date
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]

    test_data, freq, unit, freq_delta = load_test_data(pred_length, context, quantiles, dataset, forecast_date)
    run_model(test_data, quantiles, pred_length, unit, freq, freq_delta, args.save_dir, args.ckpt_path, context)