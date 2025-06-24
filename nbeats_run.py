from collections import defaultdict
import pandas as pd
import numpy as np

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS

from neuralforecast.losses.pytorch import DistributionLoss, MAE, MQLoss
from utils.evals import *
from eval_run import load_results

import argparse
import os
import json
import time
from gluonts.dataset.pandas import PandasDataset
import logging

class IgnorePLFilter(logging.Filter):
    def filter(self, record):
        keywords = ['available:', 'CUDA', 'LOCAL_RANK:', 'SLURM']
        return not any(keyword in record.getMessage() for keyword in keywords)
    
logging.getLogger('pytorch_lightning.utilities.rank_zero').addFilter(IgnorePLFilter())
logging.getLogger('pytorch_lightning.accelerators.cuda').addFilter(IgnorePLFilter())
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 128  # batch size: any positive integer
TEST = 100  # test set length: any positive integer

DEFAULT_CONFIG = {
        "max_steps": 100,
        "val_check_steps": 10,
        "learning_rate": 0.001,
        "early_stop_patience_steps": 2,
        "n_blocks": (1,1,1),
        "loss_function": "MQLoss()"
    }

def eval_wql(save_dir):
    _, dataset, model = save_dir.split('/')
    data_df, results_df, freq_delta, quantiles_dict = load_results(dataset, model)
    wql_avg, wql_arr = wql(quantiles_dict, data_df, freq_delta)
    return wql_avg

def run_model(dataset_fn, quantiles, pred_length, unit, freq, freq_delta, save_dir, context_length, forecast_date, config=DEFAULT_CONFIG):
    os.makedirs(save_dir, exist_ok=True)


    # Load dataframe and GluonTS dataset
    df_all = pd.read_csv(dataset_fn, index_col=0, parse_dates=['ds'])    
    ds = PandasDataset.from_long_dataframe(df_all, target="y", item_id="unique_id", timestamp='ds')
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
        forecast_date = min(df_all['ds']) + context_length * freq_delta
    else:
        forecast_date = pd.Timestamp(forecast_date)
    end_date = max(df_all['ds'])
    
    unique_ids = df_all['unique_id'].unique()
    for unique_id in unique_ids:
        df = df_all.loc[df_all['unique_id']==unique_id]
        train_df = df.loc[df['ds'] <= forecast_date]
        
        loss_functions = {"MQLoss()": MQLoss(level=quantiles[len(quantiles)//2:]),
                         "DistributionLoss()": DistributionLoss(distribution='Normal', level=quantiles[len(quantiles)//2:])}
        
        model = NBEATS(h=pred_length, input_size=context_length,
                        max_steps=config['max_steps'],
                        learning_rate=config['learning_rate'],
                        early_stop_patience_steps=config['early_stop_patience_steps'],
                        val_check_steps=config['val_check_steps'],
                        n_blocks=config['n_blocks'],
                        loss=loss_functions[config['loss_function']],
                        batch_size=BSZ)
        
        model.trainer_kwargs['logger'] = False
        model.trainer_kwargs['enable_progress_bar'] = False
        fcst = NeuralForecast(
            models=[model],
            freq=freq
        )
        
        start_time = time.time()
        fcst.fit(df=train_df, val_size=pred_length*2)
        print(f'Finshed fitting in {time.time()-start_time:.2f}')
        
        forecast_cols = ["NBEATS-median",  \
                            *[f"NBEATS-lo-{quantile}" for quantile in quantiles[len(quantiles)//2:]], \
                            *[f"NBEATS-hi-{quantile}" for quantile in quantiles[len(quantiles)//2:]]]
        file_names = ["median_preds", \
                            *[f"quantile_{100-quantile}_preds" for quantile in quantiles[len(quantiles)//2:]], \
                            *[f"quantile_{quantile}_preds" for quantile in quantiles[len(quantiles)//2:]]] 
        model_results = defaultdict(list)
        date_range = forecast_date + np.arange((end_date.to_period(freq) - forecast_date.to_period(freq)).n//unit_num + 1) * freq_delta
        for i, last_observed in enumerate(date_range):
            if i%100 == 0:
                print(f"Run Time: {time.time()-start_time:.2f}, date: {last_observed}")
            forecast_df = fcst.predict(df=(df.loc[df['ds']<=last_observed]))
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
            if unique_id != unique_ids[0]:
                forecast_result.to_csv(f"{save_dir}/temp.csv")
                forecast_result = pd.read_csv(f"{save_dir}/temp.csv", index_col=0, parse_dates=['ds'])
                old_result = pd.read_csv(f"{save_dir}/{file_name}.csv", index_col=0, parse_dates=['ds'])
                forecast_result = pd.concat([old_result, forecast_result], ignore_index=True)
            forecast_result.to_csv(f"{save_dir}/{file_name}.csv")
        print(f"Unqiue_id: {unique_id} in {time.time()-start_time:.2f}")

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
        "--config_file", type=str, default=None, help="Model parameter config file {default is grid search}"
    )

    args = parser.parse_args()
    PDT = args.pred_length
    CTX = args.context
    quantiles = [int(quantile) for quantile  in args.quantiles.split(',')]
    
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        run_model(args.dataset, quantiles, PDT, None, None, None, args.save_dir, CTX, args.forecast_date, config=config)
    else:
        # Grid Search
        all_configs = {
            "max_steps": [100, 1000],
            "learning_rate": [0.001, 0.0001],
            "early_stop_patience_steps": [-1, 2],
            "n_blocks": [(1,1,1), (3,3,3)],
            "val_check_steps": [10, 100],
            "loss_function": ["DistributionLoss()", "MQLoss()"]
        }

        best_config = {}
        best_wql = float('inf')
        config = {}
        for max_steps in all_configs['max_steps']:
            config['max_steps'] = max_steps
            config['val_check_steps'] = max_steps // 10
            for learning_rate in all_configs['learning_rate']:
                config['learning_rate'] = learning_rate
                for early_stop_patience_steps in all_configs['early_stop_patience_steps']:
                    config['early_stop_patience_steps'] = early_stop_patience_steps
                    for n_blocks in all_configs['n_blocks']:
                        config['n_blocks'] = n_blocks
                        for loss_function in all_configs['loss_function']:
                            config['loss_function'] = loss_function
                            run_model(args.dataset, quantiles, PDT, None, None, None, args.save_dir, CTX, args.forecast_date, config=config)
                            wql_result = eval_wql(args.save_dir)
                            print(f"Ran: {config} with score: {wql_result}")
                            if wql_result < best_wql:
                                best_config = config.copy()
                                best_wql = wql_result


        run_model(args.dataset, quantiles, PDT, None, None, None, args.save_dir, CTX, args.forecast_date, config=best_config)
        print(f"wql_avg: {eval_wql(args.save_dir)}")
        print(f"best_config: {best_config}")
        with open(f"{args.save_dir}/best_config.json", 'w') as file:
            json.dump(best_config, file, indent=4)
    