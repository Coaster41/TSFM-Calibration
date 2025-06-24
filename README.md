# Calibration Properties of Time Series Foundation Models
This is the official code repository for the paper "Calibration Properties of Time Series Foundation Models" 
Authors: Coen Adler, Yuxin Chang, Samar Abdi, Padhraic Smyth

## Prerequisites
To install required packages for the foundation models follow instructions on the original code repositories: ([MOIRAI](https://github.com/SalesforceAIResearch/uni2ts), [TimesFM](https://github.com/google-research/timesfm), [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama), [Chronos/Chronos-Bolt](https://github.com/amazon-science/chronos-forecasting/tree/main))

AutoARIMA and N-BEATS can be installed using Nixtla's forecasting packages: `pip install statsforecast` and `pip install neuralforecast`

## Datasets
Datasets must be installed manually through their official sites with preprocessing steps provided in the [data_prep](data_prep) folder

## Running Models
Use the `{model}_run.py` files to get model predictions on each dataset. Sample command: `python timesfm_run.py --dataset "data/patents/y_patents.csv" --save_dir "model_results/patents/timesfm"  --pred_length 48 --context 128 --forecast_date "2004-01-01 00:00:00" --quantiles "10,20,30,40,60,70,80,90"`

To evaluate the models use `eval_run.py` which computes the various metrics used in the paper. Note: this code is not optimized and may take upwards of an hour 