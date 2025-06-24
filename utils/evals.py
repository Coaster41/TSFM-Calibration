import numpy as np
import pandas as pd


def mae(results_df, data_df, freq_delta):
    # Mean Absolute Error
    pred_length = int(results_df.columns[-1])
    mae_arr = []
    for h in range(1,pred_length+1):
        shift_results = results_df[['ds', 'unique_id', str(h)]]
        shift_results.loc[:,'ds'] += freq_delta * h
        merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
        mean_abs_error = np.mean(np.abs(merged_results['y'] - merged_results[str(h)]))
        mae_arr.append(mean_abs_error)
    return np.mean(mae_arr), np.array(mae_arr)


def mase(results_df, data_df, freq_delta):
    # Mean Absolute Scaled Error
    pred_length = int(results_df.columns[-1])
    mae_arr = []
    for h in range(1,pred_length+1):
        shift_results = results_df[['ds', 'unique_id', str(h)]]
        shift_results.loc[:,'ds'] += freq_delta * h
        merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
        mean_abs_error = np.mean(np.abs(merged_results['y'] - merged_results[str(h)]))
        mae_arr.append(mean_abs_error)
    
    # naive mae
    shift_results = data_df.copy()
    shift_results.loc[:, 'ds'] += freq_delta
    shift_results = shift_results.rename(columns={"y": "1"})
    merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
    mae_n = np.mean(np.abs(merged_results['y'] - merged_results["1"]))
    return np.mean(mae_arr) / mae_n, np.array(mae_arr) / mae_n


def tce(lower_df, upper_df, data_df, freq_delta, confidence):    
    # Tailed Calibration Error
    pred_length = int(lower_df.columns[-1])
    outside_ratio = (1-confidence)/2
    tce_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        mean_upper_outside = np.mean(merged_upper['y'] > merged_upper[str(h)])
        mean_lower_outside = np.mean(merged_lower['y'] < merged_lower[str(h)])
        tce_arr.append(abs(outside_ratio - mean_upper_outside) + abs(outside_ratio - mean_lower_outside))
    return np.mean(tce_arr)/2, np.array(tce_arr)/2

def pce(quantiles_dict, data_df, freq_delta):    
    # Probabilistic Calibration Error
    pce_arr = []
    for quantile, quantile_df in quantiles_dict.items():
        pred_length = int(quantile_df.columns[-1])
        ce_arr = []
        for h in range(1,pred_length+1):
            shift_quantile = quantile_df[['ds', 'unique_id', str(h)]]
            shift_quantile.loc[:,'ds'] += freq_delta * h
            merged_quantile = pd.merge(data_df, shift_quantile, on=['unique_id', 'ds'], how='inner')
            mean_quantile_inside = np.mean(merged_quantile['y'] <= merged_quantile[str(h)])
            ce_arr.append(abs(quantile - mean_quantile_inside))
        pce_arr.append(ce_arr)
    pce_arr = np.mean(pce_arr, axis=0)
    return np.mean(pce_arr), pce_arr

def wql(quantiles_dict, data_df, freq_delta):
    # Weighted Quantile Loss
    ql_arr = []
    for quantile, quantile_df in quantiles_dict.items():
        quantile_ql_arr = []
        pred_length = int(quantile_df.columns[-1])
        for h in range(1,pred_length+1):
            shift_results = quantile_df[['ds', 'unique_id', str(h)]]
            shift_results.loc[:,'ds'] += freq_delta * h
            merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
            quantile_loss = np.sum((2*(1-quantile)*(merged_results[str(h)] - merged_results['y'])*(merged_results[str(h)] >= merged_results['y'])) \
                            + (2*(quantile)*(merged_results['y'] - merged_results[str(h)])*(merged_results[str(h)] < merged_results['y'])))
            quantile_ql_arr.append(quantile_loss)
        ql_arr.append(quantile_ql_arr)

    scale = np.sum(merged_results['y'])
    wql_arr = np.array(ql_arr) / scale
    return np.sum(wql_arr), np.sum(wql_arr, axis=0)


def msis(lower_df, upper_df, data_df, freq_delta, confidence):  
    # Mean Scaled Interval Score  
    pred_length = int(lower_df.columns[-1])
    mis_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        mean_interval_score = np.mean( (merged_upper[str(h)] - merged_lower[str(h)]) \
                                      + 2/(1-confidence) * (merged_lower[str(h)] - merged_lower['y']) * (merged_lower['y'] < merged_lower[str(h)]) \
                                      + 2/(1-confidence) * (merged_upper['y'] - merged_upper[str(h)]) * (merged_upper['y'] > merged_upper[str(h)]) )
        mis_arr.append(mean_interval_score)
    
    # naive mae
    shift_results = data_df.copy()
    shift_results.loc[:, 'ds'] += freq_delta
    shift_results = shift_results.rename(columns={"y": "1"})
    merged_results = pd.merge(data_df, shift_results, on=['unique_id', 'ds'], how='inner')
    mae_n = np.mean(np.abs(merged_results['y'] - merged_results["1"]))

    return np.mean(mis_arr) / mae_n, np.array(mis_arr) / mae_n

def msiw(lower_df, upper_df, data_df):
    # Mean Scaled Interval Width - Scaled by MAE
    pred_length = int(lower_df.columns[-1])
    mis_arr = []
    for h in range(1,pred_length+1):
        mis = np.mean((upper_df[str(h)] - lower_df[str(h)]))
        mis_arr.append(mis)
    
    # naive mae
    scale = np.mean(pd.merge(data_df, upper_df, on=['unique_id', 'ds'], how='inner')['y'])
    return np.mean(mis_arr) / scale, np.array(mis_arr) / scale

def siw_mean(quantiles_dict, data_df, confidences):
    # Scaled Interval Width (Mean) - Scaled by True Quantiles
    pred_length = int(quantiles_dict[0.5].columns[-1])
    merged_results = pd.merge(data_df, quantiles_dict[0.5][['ds', 'unique_id', '1']], on=['unique_id', 'ds'], how='inner')
    siw_arr = []
    h = [str(i) for i in range(1,pred_length+1)]
    for confidence in confidences:
        if confidence == 0:
            continue
        upper_df = quantiles_dict[round(0.5 + confidence/2, 1)]
        lower_df = quantiles_dict[round(0.5 - confidence/2, 1)]
        iqr = upper_df[h] - lower_df[h]
        iqr /= (np.quantile(merged_results['y'], q=round(0.5 + confidence/2, 1)) \
                - np.quantile(merged_results['y'], q=round(0.5 - confidence/2, 1)))
        siw_arr.append(np.mean(iqr, axis=0))
    return np.mean(siw_arr), np.mean(siw_arr, axis=0)


def siw(lower_df, upper_df, data_df, confidence):
    # Scaled Interval Width - Scaled by True Quantiles
    pred_length = int(upper_df.columns[-1])
    merged_results = pd.merge(data_df, upper_df[['ds', 'unique_id', '1']], on=['unique_id', 'ds'], how='inner')
    h = [str(i) for i in range(1,pred_length+1)]
    iqr = upper_df[h] - lower_df[h]
    iqr /= (np.quantile(merged_results['y'], q=round(0.5 + confidence/2, 1)) \
            - np.quantile(merged_results['y'], q=round(0.5 - confidence/2, 1)))
    siw_arr = np.mean(iqr, axis=0)
    return np.mean(siw_arr), siw_arr.to_numpy().flatten()

def cce(lower_df, upper_df, data_df, freq_delta, confidence):    
    # Centered Calibration Error
    pred_length = int(lower_df.columns[-1])
    cce_arr = []
    for h in range(1,pred_length+1):
        shift_lower = lower_df[['ds', 'unique_id', str(h)]]
        shift_lower.loc[:,'ds'] += freq_delta * h
        shift_upper = upper_df[['ds', 'unique_id', str(h)]]
        shift_upper.loc[:,'ds'] += freq_delta * h
        merged_upper = pd.merge(data_df, shift_upper, on=['unique_id', 'ds'], how='inner')
        merged_lower = pd.merge(data_df, shift_lower, on=['unique_id', 'ds'], how='inner')
        middle = (merged_upper['y'] <= merged_upper[str(h)]) & (merged_lower['y'] >= merged_lower[str(h)])
        mean_middle = np.mean(middle)
        cce_arr.append(confidence - mean_middle)
    return np.mean(cce_arr), np.array(cce_arr)

def stce(quantile_df, data_df, freq_delta, quantile):
    # Single Tailed Calibration Error
    pred_length = int(quantile_df.columns[-1])
    tce_arr = []
    for h in range(1,pred_length+1):
        shift_quantile = quantile_df[['ds', 'unique_id', str(h)]]
        shift_quantile.loc[:,'ds'] += freq_delta * h
        merged_quantile = pd.merge(data_df, shift_quantile, on=['unique_id', 'ds'], how='inner')
        if quantile >= 0.5:
            mean_quantile_outside = np.mean(merged_quantile['y'] > merged_quantile[str(h)])
            tce_arr.append(mean_quantile_outside - (1-quantile)) # over confidence: outside is greater
        else:
            mean_quantile_outside = np.mean(merged_quantile['y'] < merged_quantile[str(h)])
            tce_arr.append(mean_quantile_outside - (quantile)) # over confidence: outside is greater
    return np.mean(tce_arr), np.array(tce_arr)
