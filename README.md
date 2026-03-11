# Hybrid Forecast Aggregation

## Overview

This project analyzes and compares multiple crowd forecast aggregation techniques using data from the **HFC RCTA Forecasting Tournament**. The goal is to evaluate how different aggregation strategies perform in terms of predictive accuracy and to propose an improved hybrid method that incorporates forecaster skill.

Crowd forecasting platforms often rely on aggregating predictions from many participants. Choosing the right aggregation method can significantly influence prediction quality. This project evaluates several classical aggregation approaches and introduces a **skill-weighted hybrid aggregation method** designed to improve overall forecasting performance.

---

## Dataset

The analysis uses publicly available data from the **HFC RCTA forecasting tournament**.

The dataset consists of three main files:

1. **Daily Forecasts**

   * Contains the most recent forecast made by each forecaster before the daily scoring cutoff.
   * Includes identifiers for questions, answers, and predictors.
   * This file serves as the primary dataset for aggregation.

2. **Questions and Answers**

   * Contains metadata about each question.
   * Includes resolved outcomes and information about whether a question uses ordinal scoring.

3. **Prediction Sets**

   * A log of individual forecasts submitted during the lifetime of each question.
   * Includes metadata such as submission timing and forecaster identifiers.

Due to file size limitations, the dataset is **not included in this repository**.

You can download the dataset from:

```
https://dataverse.harvard.edu/dataverse/hfc
```

After downloading, place the files in the project root directory.

---

## Data Preprocessing

Before aggregation, several preprocessing steps are applied:

* Removal of **default forecasts** (system-generated base rate predictions)
* Parsing and validation of timestamps
* Conversion of identifiers to categorical types for memory efficiency
* Handling missing or malformed values
* Rounding forecast probabilities to avoid floating-point noise

Removing default forecasts reduced the dataset size by approximately **22%**, ensuring that only meaningful predictions were included in the analysis.

---

## Aggregation Methods Evaluated

The following aggregation techniques were implemented and evaluated:

* **Raw Mean**
* **Trimmed Mean**
* **Median**
* **Geometric Mean**
* **Geometric Mean of Odds**

Forecast accuracy is evaluated using the **Brier Score**, which measures the squared difference between predicted probabilities and actual outcomes.

Lower Brier scores indicate better predictive performance.

---

## Results

| Method                 | Average Brier Score |
| ---------------------- | ------------------- |
| Raw Mean               | 0.122722            |
| Trimmed Mean           | 0.123196            |
| Median                 | 0.125036            |
| Geometric Mean of Odds | 0.129864            |
| Geometric Mean         | 0.136829            |

Arithmetic-based aggregation methods performed best because they balance the crowd’s uncertainty and reduce the influence of extreme or overconfident forecasts.

---

## Hybrid Skill-Weighted Aggregation

To improve forecasting accuracy, a **skill-weighted aggregation method** was implemented.

In this approach:

* Each forecaster is assigned a weight based on their **historical prediction accuracy**.
* Accuracy is measured using the **inverse of the average Brier score** computed from past resolved questions.
* Forecasts are aggregated using a **weighted mean**, giving more influence to historically accurate forecasters.

This hybrid approach improves calibration and reduces overall Brier error compared to baseline methods.

---

## Project Structure

```
hybrid-forecast-aggregation/
│
├── main.py        # Forecast aggregation pipeline
├── memo.pdf       # Detailed project analysis
└── README.md      # Project documentation
```

---

## Requirements

The project uses standard Python data science libraries:

* numpy
* pandas

Install dependencies with:

```
pip install numpy pandas
```

---

## How to Run

1. Download the dataset.
2. Place the dataset files in the project directory.
3. Run:

```
python main.py
```

The script will:

* preprocess the data
* compute aggregated forecasts
* evaluate performance using Brier scores
* output a comparison of aggregation methods.

---

## Key Takeaways

* Simple aggregation methods like the **raw mean** often perform surprisingly well in crowd forecasting.
* Arithmetic-based aggregation tends to outperform geometric methods due to better handling of forecast uncertainty.
* Incorporating **forecaster skill** through weighted aggregation can further improve predictive accuracy.

---
