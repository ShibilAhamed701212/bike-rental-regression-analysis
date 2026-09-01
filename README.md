# Seoul Bike Sharing Demand — Regression Analysis

A machine learning project that predicts bike rental demand in Seoul using regression techniques. Built with Python in a Jupyter Notebook environment.

## Overview

This project analyzes the [Seoul Bike Sharing Demand dataset](http://archive.ics.uci.edu/ml) from the UCI Machine Learning Repository to predict the number of bikes rented at noon based on weather and environmental features.

## Dataset

- **Source:** [Seoul Open Data](http://data.seoul.go.kr/) & [South Korea Public Holidays](https://publicholidays.go.kr)
- **File:** `seoul+bike+sharing+demand/SeoulBikeData.csv`
- **Features used:** Temperature, Humidity, Dew Point Temperature, Solar Radiation, Rainfall, Snowfall
- **Target:** Bike rental count at noon (`bike_count`)

## Approach

1. **Data Loading & Cleaning** — Load the CSV, drop date/holiday/season columns, filter to noon-hour records.
2. **Exploratory Data Analysis** — Scatter plots of each feature vs. bike count to identify correlations.
3. **Feature Selection** — Drop low-signal features (`wind`, `visibility`, `functional`).
4. **Train / Validation / Test Split** — 60 / 20 / 20 random split.
5. **Modeling**
   - **Linear Regression** (scikit-learn) — single-feature (temperature) and multi-feature models.
   - **Neural Network** (TensorFlow / Keras) — deeper regression model for comparison.

## Tech Stack

| Tool | Purpose |
|---|---|
| **pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **scikit-learn** | Linear Regression, preprocessing |
| **TensorFlow / Keras** | Neural network regression |
| **imbalanced-learn** | Oversampling utilities |

## Getting Started

```bash
# Clone the repo
git clone https://github.com/ShibilAhamed701212/bike-rental-regression-analysis.git
cd bike-rental-regression-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow imbalanced-learn

# Run the notebook
jupyter notebook fcc_bikes_regression.ipynb
```

## Results

- **Linear Regression (temperature only):** R² ≈ 0.41 on test set
- **Multi-feature & neural network models** improve upon the baseline — see the notebook for full results and visualizations.

## License

This project is for educational purposes. Dataset provided under the [UCI ML Repository](http://archive.ics.uci.edu/ml) terms.
