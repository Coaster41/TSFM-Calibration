import pandas as pd
import numpy as np
import time
from utils.evals import *
from collections import defaultdict

UNIT_DICT = {"amazon-google": "H", "m5": "D", "glucose": "T", "meditation": "S", "amazon": "H", "patents": "M", "police": "D",
             "m5_small": "D", "meditation_small": "S", "glucose_small": "T", "patents_small": "M", "police_small": "D"}
UNIT_NUM_DICT = {"amazon-google": 1, "m5": 1, "glucose": 5, "meditation": 1, "amazon": 1, "patents": 1, "police": 1,
                 "m5_small": 1, "meditation_small": 1, "glucose_small": 5, "patents_small": 1, "police_small": 1}

def load_results(dataset, model):
    unit = UNIT_DICT[dataset]
    unit_num = UNIT_NUM_DICT[dataset]
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

    results_fn = f"model_results/{dataset}/{model}/median_preds.csv"
    data_fn = f"data/{dataset}/y_{dataset}.csv"
    results_df = pd.read_csv(results_fn, index_col=0, parse_dates=['ds'])
    data_df = pd.read_csv(data_fn, index_col=0, parse_dates=['ds'])
    if unit == 'M':
        freq_delta = pd.DateOffset(months=unit_num)
    else:
        freq_delta = pd.Timedelta(unit_num, unit)
        
    quantiles_df = []
    for quantile in quantiles:
        quantile_fn = f"model_results/{dataset}/{model}/quantile_{round(quantile * 100)}_preds.csv"
        quantiles_df.append(pd.read_csv(quantile_fn, index_col=0, parse_dates=['ds']))
    quantiles_df.insert(len(quantiles_df)//2, results_df)
    quantiles.insert(len(quantiles)//2, 0.5)
    quantiles_dict = dict(zip(quantiles, quantiles_df))
    return data_df, results_df, freq_delta, quantiles_dict


if __name__ == "__main__":
    datasets = ["amazon-google", "m5", "glucose", "meditation", "police", "patents"]
    models = ["timesfm", "moirai", "chronos", "chronos-bolt", "lag-llama", "nbeats", "autoarima"]
    results = []
    confidences = [0.8, 0.6, 0.4, 0.2]
    start_time = time.time()
    max_ids = 0
    append = True
    metrics = ['mase', 'wql', 'pce', 'siw', 'msis', 'tce', 'cce']

    for dataset in datasets:
        for model in models:
            print(f"Time: {time.time()-start_time:.4f}\t{dataset} {model}")
            data_df, results_df, freq_delta, quantiles_dict = load_results(dataset, model)
            if 'mase' in metrics:
                mase_avg, mase_arr = mase(results_df, data_df, freq_delta)
                results.append([dataset, model, 'mase', mase_avg, *mase_arr])
            if 'wql' in metrics:
                wql_avg, wql_arr = wql(quantiles_dict, data_df, freq_delta)
                results.append([dataset, model, 'wql', wql_avg, *wql_arr])
            if 'pce' in metrics:
                pce_avg, pce_arr = pce(quantiles_dict, data_df, freq_delta)
                results.append([dataset, model, 'pce', pce_avg, *pce_arr])
            if 'siw' in metrics:
                siw_avg, siw_arr = siw_mean(quantiles_dict, data_df, confidences)
                results.append([dataset, model, 'siw', siw_avg, *siw_arr])
            cce_avg_arr, cce_arr_arr = list(), list()
            for confidence in confidences:
                upper_df = quantiles_dict[round(0.5 + confidence/2, 1)]
                lower_df = quantiles_dict[round(0.5 - confidence/2, 1)]
                if 'tce' in metrics:
                    tce_avg, tce_arr = tce(lower_df, upper_df, data_df, freq_delta, confidence)
                    results.append([dataset, model, f'tce_{round(confidence*100)}', tce_avg, *tce_arr])
                if 'cce' in metrics:
                    cce_avg, cce_arr = cce(lower_df, upper_df, data_df, freq_delta, confidence)
                    results.append([dataset, model, f'cce_{round(confidence*100)}', cce_avg, *cce_arr])
                    cce_avg_arr.append(cce_avg)
                    cce_arr_arr.append(cce_arr)
                if 'msiw' in metrics:
                    msiw_avg, msiw_arr = msiw(lower_df, upper_df, data_df)
                    results.append([dataset, model, f'msiw_{round(confidence*100)}', msiw_avg, *msiw_arr])
                if 'msis' in metrics:
                    msis_avg, msis_arr = msis(lower_df, upper_df, data_df, freq_delta, confidence)
                    results.append([dataset, model, f'msis_{round(confidence*100)}', msis_avg, *msis_arr])
                if 'siw' in metrics:
                    siw_avg, siw_arr = siw(lower_df, upper_df, data_df, confidence)
                    results.append([dataset, model, f'siw_{round(confidence*100)}', siw_avg, *siw_arr])
                if 'stce' in metrics:
                    stce_avg, stce_arr = stce(lower_df, data_df, freq_delta, round(0.5 - confidence/2, 1))
                    results.append([dataset, model, f'stce_{round((0.5 - confidence/2)*100)}', stce_avg, *stce_arr])
                    stce_avg, stce_arr = stce(upper_df, data_df, freq_delta, round(0.5 + confidence/2, 1))
                    results.append([dataset, model, f'stce_{round((0.5 + confidence/2)*100)}', stce_avg, *stce_arr])
            if 'cce' in metrics:                
                results.append([dataset, model, f'cce', np.mean(cce_avg_arr), *np.mean(cce_arr_arr, axis=0)])
            df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', 'avg_result', *[str(h) for h in range(1,49)]])
            df.to_csv('model_results/metric_results_test1.csv')
    if append:
        df = pd.read_csv('model_results/metric_results.csv', index_col=0)
        df = pd.concat([df, pd.DataFrame(results, columns=['dataset', 'model', 'metric', 'avg_result', *[str(h) for h in range(1,49)]])], ignore_index=True)
    else:
        df =  pd.DataFrame(results, columns=['dataset', 'model', 'metric', 'avg_result', *[str(h) for h in range(1,49)]])
    df.to_csv('model_results/metric_results.csv')

    # groupby unique_id
    results = []
    for dataset in datasets:
        for model in models:
            load_time = time.time()
            print(f"Time-Series: {time.time()-start_time:.4f}\t{dataset} {model}")
            data_df, results_df, freq_delta, quantiles_dict = load_results(dataset, model)
            grouped_data_df = data_df.groupby('unique_id')
            grouped_results_df = results_df.groupby('unique_id')
            grouped_quantiles_dict = defaultdict(dict)
            for quantile, quantile_df in quantiles_dict.items():
                grouped_quantile_df = quantile_df.groupby('unique_id')
                for unique_id, grouped_q in grouped_quantile_df:
                    grouped_quantiles_dict[unique_id][quantile] = grouped_q
            max_ids = max(max_ids, len(grouped_data_df))

            # mase_arr, tce_arr, msis_arr, wql_arr, pce_arr, msiw_arr = list(), list(), list(), list(), list(), list()
            mase_arr, wql_arr, pce_arr, siw_mean_arr = list(), list(), list(), list()
            msis_arr = [[] for _ in range(len(confidences))]
            tce_arr = [[] for _ in range(len(confidences))]
            cce_arr = [[] for _ in range(len(confidences))]
            msiw_arr = [[] for _ in range(len(confidences))]
            siw_arr = [[] for _ in range(len(confidences))]
            stce_low_arr = [[] for _ in range(len(confidences))]
            stce_up_arr = [[] for _ in range(len(confidences))]
            for (unique_id, data_df), (_, results_df), (_, quantiles_dict) in \
                zip(grouped_data_df, grouped_results_df, grouped_quantiles_dict.items()):
                if 'mase' in metrics:
                    mase_avg, mase_h = mase(results_df, data_df, freq_delta)
                    mase_arr.append(mase_avg)
                if 'wql' in metrics:
                    wql_avg, wql_h = wql(quantiles_dict, data_df, freq_delta)
                    wql_arr.append(wql_avg)
                if 'pce' in metrics:
                    pce_avg, pce_h = pce(quantiles_dict, data_df, freq_delta)
                    pce_arr.append(pce_avg)
                if 'siw' in metrics:
                    siw_avg, siw_h = siw_mean(quantiles_dict, data_df, confidences)
                    siw_mean_arr.append(siw_avg)
                for i, confidence in enumerate(confidences):
                    upper_df = quantiles_dict[round(0.5 + confidence/2, 1)]
                    lower_df = quantiles_dict[round(0.5 - confidence/2, 1)]
                    if 'tce' in metrics:
                        tce_avg, tce_h = tce(lower_df, upper_df, data_df, freq_delta, confidence)
                        tce_arr[i].append(tce_avg)
                    if 'cce' in metrics:
                        cce_avg, cce_h = cce(lower_df, upper_df, data_df, freq_delta, confidence)
                        cce_arr[i].append(cce_avg)
                    if 'msis' in metrics:
                        msis_avg, msis_h = msis(lower_df, upper_df, data_df, freq_delta, confidence)
                        msis_arr[i].append(msis_avg)
                    if 'msiw' in metrics:
                        msiw_avg, msiw_h = msiw(lower_df, upper_df, data_df)
                        msiw_arr[i].append(msiw_avg)
                    if 'siw' in metrics:
                        siw_avg, siw_h_ = siw(lower_df, upper_df, data_df, confidence)
                        siw_arr[i].append(siw_avg)
                    if 'stce' in metrics:
                        stce_avg, stce_arr = stce(lower_df, data_df, freq_delta, round(0.5 - confidence/2, 1))
                        stce_low_arr[i].append(stce_avg)
                        stce_avg, stce_arr = stce(upper_df, data_df, freq_delta, round(0.5 + confidence/2, 1))
                        stce_up_arr[i].append(stce_avg)
            if 'mase' in metrics:
                results.append([dataset, model, 'mase', *mase_arr])
            if 'wql' in metrics:
                results.append([dataset, model, 'wql', *wql_arr])
            if 'pce' in metrics:
                results.append([dataset, model, 'pce', *pce_arr])
            if 'siw' in metrics:
                results.append([dataset, model, 'siw', *siw_mean_arr])
                results.extend([[dataset, model, f'siw_{round(confidence*100)}', *siw_arr[i]] for i, confidence in enumerate(confidences)])
            if 'tce' in metrics:
                results.extend([[dataset, model, f'tce_{round(confidence*100)}', *tce_arr[i]] for i, confidence in enumerate(confidences)])
            if 'cce' in metrics:
                results.extend([[dataset, model, f'cce_{round(confidence*100)}', *cce_arr[i]] for i, confidence in enumerate(confidences)])
                results.append([dataset, model, 'cce', *np.mean(cce_arr, axis=0)])
            if 'msis' in metrics:
                results.extend([[dataset, model, f'msis_{round(confidence*100)}', *msis_arr[i]] for i, confidence in enumerate(confidences)])
            if 'msiw' in metrics:
                results.extend([[dataset, model, f'msiw_{round(confidence*100)}', *msiw_arr[i]] for i, confidence in enumerate(confidences)])
            if 'stce' in metrics:
                results.extend([[dataset, model, f'stce_{round((0.5 - confidence/2)*100)}', *stce_low_arr[i]] for i, confidence in enumerate(confidences)])
                results.extend([[dataset, model, f'stce_{round((0.5 + confidence/2)*100)}', *stce_up_arr[i]] for i, confidence in enumerate(confidences)])

    if append:
        df = pd.read_csv('model_results/metric_results_unique_id.csv', index_col=0)
        df = pd.concat([df, pd.DataFrame(results, columns=['dataset', 'model', 'metric', *[str(h) for h in range(1,max_ids+1)]])], ignore_index=True)
    else:
        df = pd.DataFrame(results, columns=['dataset', 'model', 'metric', *[str(h) for h in range(1,max_ids+1)]])
    df.to_csv('model_results/metric_results_unique_id.csv')