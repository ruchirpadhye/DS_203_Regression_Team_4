# DS 203 Project: Forecasting U.S. Inflation with Regression
 
This is our group project for DS 203 (Intro to Data Science). The goal is to see if we can predict U.S. inflation a few months into the future using regular macroeconomic data, and to compare three regression models we learned in class: **Linear Regression, Ridge, and Lasso**.
 
**Group members:** Abhimanyu Kashyap, Ruchir Padhye, Harshil Rathi
 
---
 
## What's the question?
 
Can we use economic indicators from this month (unemployment, oil prices, interest rates, etc.) to predict what inflation will be **n months from now**?
 
We tried four forecast horizons: **n = 1, 3, 6, and 12 months ahead**.
 
Inflation is calculated from the Consumer Price Index (CPI) as the year-over-year change:
 
```
Inflation_t = 100 * (CPI_t / CPI_(t-12) - 1)
```
 
and the target for each horizon is just `Inflation_(t+n)`. So for the 3-month model, we use data from month t to predict inflation at month t+3.
 
---
 
## Data
 
All the data comes from **FRED** (Federal Reserve Economic Data). we downloaded 11 series as CSVs and put them in a `Datasets/` folder:
 
| FRED ID | What it is |
|---|---|
| `CPIAUCSL` | Consumer Price Index (used to make inflation) |
| `T10Y2Y` | 10-year minus 2-year Treasury yield spread (daily) |
| `UNRATE` | Unemployment rate |
| `INDPRO` | Industrial production |
| `FEDFUNDS` | Federal funds rate |
| `PAYEMS` | Total nonfarm payrolls |
| `PPIACO` | Producer price index |
| `HOUST` | Housing starts |
| `DCOILWTICO` | WTI crude oil price (daily) |
| `M2SL` | M2 money supply |
| `UMCSENT` | U of Michigan consumer sentiment |

---
 
## How the project works
 
The notebook follows this pipeline:
 
**Data Cleaning → Feature Engineering → Chronological Split → Preprocessing → Model Training → Evaluation**
 
### 1. Data cleaning
- Two of the series (`T10Y2Y` and oil) are daily, but the rest are monthly. we took the **monthly average** of the daily ones so everything lines up.
- FRED sometimes uses `.` for missing values, which breaks the data types, so we used `pd.to_numeric(..., errors='coerce')` to turn those into NaN.
- Since the series start at different times (oil starts in 1986, for example), we only kept the date range where **all 11** series have data.
### 2. Feature engineering
A lot of the raw series (like CPI, industrial production, M2) just keep going up over time, and regressing on those can give fake/spurious relationships. So we converted them:
- **Year-over-year growth rates** for `INDPRO`, `PAYEMS`, `PPIACO`, `DCOILWTICO`, `M2SL`, and `HOUST`
- **Monthly change in unemployment** (`d_unrate`), in addition to the unemployment level
- Kept `T10Y2Y`, `FEDFUNDS`, `UNRATE`, and `UMCSENT` as they are, since they're already rates/spreads/indexes
- Added **current inflation** as a feature too, because inflation tends to stick around (it's autocorrelated)
This gives **12 features** in total.
 
### 3. EDA (exploring the data)
- We plotted inflation over time and counted how many months were deflation, low, moderate, or high inflation
- Looked at skewness, outliers (using the 1.5 × IQR rule), and the spread of each feature
- Checked correlations between each feature and current inflation, then **lead-lag correlations** (feature at time t vs inflation at t+1, t+3, t+6, t+12) shown as a heatmap
A few things we noticed from this part:
- The features are on really different scales (`d_unrate` is tiny, oil growth is huge), so scaling is needed before Ridge and Lasso.
- Producer prices and oil are the most related to inflation at the same time, which makes sense since costs get passed on to consumers.
- Current inflation is the strongest predictor for short horizons, but it matters less at 6 and 12 months, where things like money supply growth, the Fed funds rate, and the yield spread seem to matter more.
- we kept the outliers (like COVID in 2020) instead of deleting them, because they're real events and not mistakes.
### 4. Train / validation / test split
Since this is time series data, we **did not shuffle**. Shuffling would let the model "see the future" when training. Instead we split in order:
- First **70%** → train
- Next **15%** → validation
- Last **15%** → test
### 5. Preprocessing (without data leakage)
- Missing values are filled with the **median from the training set only**
- `StandardScaler` is fit on the training data only (it's inside a sklearn `Pipeline` so this happens automatically)
### 6. Models
| Model | Hyperparameter (alpha) values tried |
|---|---|
| Linear Regression | none |
| Ridge (L2) | 0.01, 0.1, 1, 10, 100 |
| Lasso (L1) | 0.001, 0.01, 0.05, 0.1, 0.5 |
 
For each horizon and each model, we picked the best alpha using **validation RMSE**. Then we checked the chosen model on the test set one time only.
 
### 7. Evaluation metrics
- **MAE** (mean absolute error)
- **RMSE** (root mean squared error)
- **R²**
---
 
## Results
 
The notebook prints a results table with the best alpha, validation RMSE, and test MAE / RMSE / R² for every model and horizon. It also makes:
- A line plot of **test RMSE vs forecast horizon** for all three models
- Actual vs predicted inflation plots for the test period (one per horizon)
- Tables of the standardized **Lasso and Ridge coefficients**, to see which features mattered most
*(Fill in our numbers from the output here, for example which model did best at each horizon and how much worse the 12-month forecast was compared to the 1-month one.)*
 
---
 
## How to run it

1. Install the libraries:
```
pip install numpy pandas matplotlib scikit-learn
```
2. Open `DS203_Inflation_Forecasting.ipynb` in Jupyter and run all the cells from top to bottom
**Heads up:** we wrote the file paths with a backslash (`Datasets\\{sid}.csv`) since we're on Windows. If you're on Mac or Linux, you'll probably have to change it to `Datasets/{sid}.csv` or it won't find the files.
 
---
 
## Things we'd improve / limitations
 
- The test set (the last 15%) includes the 2021–2022 inflation spike, which is really hard to predict, so the test results might look worse than they would in a calmer period.
- We only used linear models because that's what we covered in class. Something like a random forest might pick up non-linear patterns.
- The alpha grids are pretty small, so a bigger search might do slightly better.
- We used a single train/val/test split. Time series cross-validation would probably give a more reliable estimate.
- Some of the features are probably correlated with each other (PPI and oil, for example), which is partly why Ridge and Lasso are useful here.
---
 
## Tools used
 
Python, pandas, NumPy, matplotlib, scikit-learn

 
**Team:** Abhimanyu Kashyap, Ruchir Padhye, Harshil Rathi
 
