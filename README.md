# A compartmental model for neologisms: attention vs. use in language change

Data, analysis code, and figure materials accompanying the manuscript.

## Download the materials

The complete package is provided in [data_and_code.zip](data_and_code.zip). It contains 431 files and preserves the directory structure of the supplied data package.

Download and extract the archive, then open the extracted `data upload to OSF_1120` directory. Paths below are relative to that directory.

## Package contents

| Directory or file | Contents |
| --- | --- |
| `Raw Data/Google Trends/` | 190 CSV files containing exported search-interest time series. |
| `Raw Data/Monitor Corpus/` | 190 CSV files containing dated word-frequency records from the Oxford Monitor Corpus of English. |
| `Raw Data/wordlist.xlsx` | Word-list spreadsheet. |
| `SIR fitting/` | SIR fitting script and documents recording fitting settings for the two data sources. |
| `Multi-model Competition/` | Batch model-comparison script, saved per-word metrics and logisticness scores, and score-recalculation utilities. |
| `Figure_sources/` | Figure-generation scripts, source data, and figure files, grouped by figure or analysis. |

## Data fields

Google Trends exports contain date labels and search-interest values, with export metadata at the beginning of the files. Monitor CSV files contain an observation date (`Time` or `time`) and a recorded word count (`count`).

The saved `v3lite_logisticness_scores.csv` table contains word identifiers, the recorded number of observations, cumulative Logistic R², test RMSE values for Logistic, ARIMA and Naive, the baseline RMSE, two score components, and the resulting logisticness score.

## Python environment

The analysis and plotting scripts use Python with NumPy, pandas, SciPy, Matplotlib, and statsmodels. Individual scripts specify their command-line options and additional requirements.

## Recalculate saved logisticness scores

Run the following from the extracted `data upload to OSF_1120` directory:

```bash
python "Multi-model Competition/code/recompute_logisticness.py" --input "Multi-model Competition/v3lite_logisticness_scores.csv" --output-dir "verification/score"
```

This utility recalculates scores from the saved per-word R² and RMSE values using equal component weights of 0.5 and 0.5. The training/test proportion of 70/30 is a separate forecasting setting.

## Package integrity

SHA-256 of `data_and_code.zip`:

```text
6dc30044e6751eab24c9e17f221b37e54b28e751d5c776fd4567c403c821250d
```
