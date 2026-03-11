# Yes Bank Stock Price Prediction

This is one of my machine learning projects where I worked on predicting Yes Bank's monthly closing stock price. I chose this dataset because Yes Bank has a really interesting story — the stock was growing steadily for over a decade, then completely collapsed after the Rana Kapoor fraud case in 2018. That kind of real-world event makes the data much more interesting to work with than a clean textbook dataset.

---

## About the Project

The dataset has 185 months of stock price data starting from July 2005. Each row has five columns — Date, Open, High, Low, and Close. My goal was to predict the Close price using the other features along with some new features I engineered myself.

After trying three different models, my final Random Forest model achieved **98.76% accuracy** with an average prediction error of just 6.80 INR.

---

## Files in this Repository

```
yes-bank-stock-prediction/
│
├── yes-bank-stock-prediction.ipynb   ← Main notebook with all code
├── data_YesBank_StockPrices.csv      ← Dataset
├── yes_bank_stock_predictor.pkl      ← Saved trained model
└── README.md
```

---

## Dataset

The dataset is pretty clean — no missing values, no duplicates. Just 185 rows and 5 columns.

| Column | What it means |
|--------|--------------|
| Date | Month and year (Jul-05 format) |
| Open | Price at start of month |
| High | Highest price that month |
| Low | Lowest price that month |
| Close | Price at end of month (this is what I'm predicting) |

Prices range from 5.55 INR all the way up to 404 INR — that wide range is entirely because of the fraud case.

---

## What I Did — Step by Step

**1. Data Wrangling**
The Date column was stored as plain text so I converted it to datetime format first. Then I extracted Year, Month, and Month Name as separate columns. I also sorted everything chronologically since this is time series data.

I checked for outliers using IQR and found 9 outliers in the Close column, but I kept all of them. They represent the actual 2017-2018 bull run peak prices — removing them would have deleted the most important period in the dataset.

**2. Exploratory Data Analysis**
I made 15 charts following the UBM rule (Univariate, Bivariate, Multivariate). The most important one was the closing price over time with a vertical red line at 2018 marking the fraud case — it tells the whole story of this stock in one chart.

A few things I noticed:
- The Low price has 0.9954 correlation with Close — incredibly strong
- Month has almost zero correlation with Close — no seasonality in this stock
- After the fraud, variance more than doubled (7706 → 16405), meaning the stock became much more unpredictable

**3. Hypothesis Testing**
I ran three statistical tests:
- T-Test to check if fraud impacted average price (failed to reject — there's a limitation with how I defined the cutoff year)
- Levene's Test to check if volatility changed — this one rejected the null hypothesis, proving volatility significantly increased after fraud
- Pearson Correlation to prove Low is a statistically significant predictor of Close — rejected with p-value essentially zero

**4. Feature Engineering**
I created a few new features:
- `Price_Spread_Pct` — how much the price moved as a percentage that month
- `Open_Close_Diff` — difference between open and close, tells you if the month ended up or down
- `Prev_Close` — previous month's closing price (lag feature)
- `Era_Encoded` — 0 for pre-fraud years, 1 for post-fraud

Then I dropped High, Open, and Prev_Close because they were all 0.98+ correlated with Low — keeping them would have caused multicollinearity. Final model used 6 features.

I applied log transformation to Low and Price_Range to fix their skewness, and StandardScaler after the train-test split to avoid data leakage.

**5. Model Training**

| Model | Accuracy | Notes |
|-------|----------|-------|
| Linear Regression | 83.71% | Baseline model |
| Linear Regression (Ridge) | 84.13% | Small improvement with regularization |
| Random Forest (base) | 98.74% | Big jump in performance |
| **Random Forest (tuned)** | **98.76%** | Best model, selected as final |
| SVR (base) | 10.53% | Very poor with default params |
| SVR (tuned) | 96.13% | Massive improvement after GridSearchCV |

The SVR result was the most interesting — default parameters gave 10.53%, but after tuning with C=100 and gamma=auto it jumped to 96.13%. That's how much hyperparameter tuning can matter.

For Random Forest I used RandomizedSearchCV instead of GridSearchCV because Random Forest has a lot of parameters and a full grid search would take too long.

**6. Feature Importance**
After training the final Random Forest model, Low price accounts for 99.5% of the feature importance. This makes sense — the lowest price a stock touches in a month is very strongly tied to where it closes.

---

## Results

My final model is **Random Forest with n_estimators=200 and max_depth=10**.

- R² = 0.9876
- MAE = 6.80 INR
- RMSE = 10.59 INR
- Accuracy = 98.76%

The model is saved as a `.pkl` file using joblib and is ready for deployment.

---

## How to Run

```bash
git clone https://github.com/ronakjha2002/yes-bank-stock-prediction.git
cd yes-bank-stock-prediction
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib
jupyter notebook yes-bank-stock-prediction.ipynb
```

To use the saved model directly:

```python
import joblib
model = joblib.load('yes_bank_stock_predictor.pkl')
predictions = model.predict(your_data)
```

---

## Libraries Used

pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, joblib

---

## What I Learned

The biggest takeaway from this project was how much a single real-world event (the 2018 fraud case) can completely change the statistical properties of a dataset. Pre-fraud and post-fraud data behave like two completely different datasets — different price ranges, different volatility, different distributions. The Era_Encoded feature was specifically created to help the model understand this structural break.

Also, SVR with default parameters is basically useless for this kind of financial data. Hyperparameter tuning took it from 10% to 96% — that was a good reminder to never trust default parameters blindly.

---

*This project was done as part of my ML learning journey. Feel free to use the code or reach out if you have questions.*
