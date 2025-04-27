# Sales Forecasting Using Temporal Fusion Transformer (TFT)

![Chart.js](https://img.shields.io/badge/chart.js-F5788D.svg?style=plastic&logo=chart.js&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=plastic&logo=fastapi) ![JWT](https://img.shields.io/badge/JWT-black?style=plastic&logo=JSON%20web%20tokens) ![Redux](https://img.shields.io/badge/redux-%23593d88.svg?style=plastic&logo=redux&logoColor=white) ![React](https://img.shields.io/badge/react-%2320232a.svg?style=plastic&logo=react&logoColor=%2361DAFB) ![Express.js](https://img.shields.io/badge/express.js-%23404d59.svg?style=plastic&logo=express&logoColor=%2361DAFB) ![TailwindCSS](https://img.shields.io/badge/tailwindcss-%2338B2AC.svg?style=plastic&logo=tailwind-css&logoColor=white) ![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=plastic&logo=vite&logoColor=white) ![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=plastic&logo=mongodb&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=plastic&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=plastic&logo=numpy&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=plastic&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=plastic&logo=scikit-learn&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=plastic&logo=plotly&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=plastic&logo=Matplotlib&logoColor=black)

---

Welcome to the **Sales Forecasting** repository, part of the Sales Prediction Application developed for **INC 2025**.

This repository contains the **Machine Learning model** built using the **Temporal Fusion Transformer (TFT)** architecture.  
The model is designed to **predict quarterly sales** based on historical distributor sales data.

## Project Overview
The primary objective of this project is to accurately forecast future sales for any given quarter by analyzing patterns and trends from previous quarters.  
This forecasting system aims to assist businesses in better demand planning and inventory management.

## Repositories
This is part of a larger project with separate components for the frontend and backend:

- **Frontend Application**: [Sales Forecasting Frontend](https://github.com/harshapeshave641/Demand-Forecasting-Frontend)
- **Backend API**: [Sales Forecasting Backend](https://github.com/harshapeshave641/Demand-Forecasting-Backend)

---

## Model Architecture

The core model is based on the **Temporal Fusion Transformer (TFT)**, a powerful deep learning model for time series forecasting.  
The architecture includes:

- **Variable Selection Networks**: Dynamically select the most relevant input variables.
- **Gated Residual Networks (GRN)**: Process information with skip connections.
- **LSTM Encoder-Decoder**: Capture sequential dependencies in time series.
- **Multi-Head Attention**: Focus on the most relevant past time steps.
- **Static Covariates Encoder**: (optional) Handle features that don't change over time.
- **Quantile Regression Loss**: Predict multiple quantiles (for probabilistic forecasting).

### Model Diagram
---
Here’s a high-level architecture overview:

<!-- ![Architecture Diagram](./images/diagram.png) -->
<center>

<img src="./images/diagram.png" title="Architecture Diagram" height="1080px"></img>

</center>


```plaintext
RAW SALES DATA
(distributor_id, sku, category, sales, quarter, year, festivals, etc.)
      ↓
⮕ Preprocessing
- Fill missing values
- Encode categorical variables
- Normalize real-valued features
      ↓
⮕ TimeSeriesDataSet (PyTorch Forecasting)
- time_idx = time index (quarter/year)
- target = sales
- group_ids = distributor_id
- known/observed/static features
      ↓
⮕ Temporal Fusion Transformer (TFT)
- Variable Selection Networks
- LSTM Encoder-Decoder
- Multi-Head Attention
- Gated Residual Networks
- Quantile Loss (probabilistic forecasts)
      ↓
⮕ Predictions
- Forecast next quarters
- Visualize quantiles
- Sales trend analysis
```

---
*Note: A detailed visualization of TFT is available in the [original TFT paper](https://arxiv.org/abs/1912.09363).*



## Model Metrics
The model achieves high accuracy in predicting quarterly trends based on previous distributor sales records.

![Prediction Metrics](./images/prediction-metrics1.png)


*Prediction Metrics display values on x-axis in ₹ and the frequency on the y-axis*

---

## Future Improvements
- Hyperparameter tuning for even better prediction accuracy
- Incorporating additional external covariates (e.g., economic indicators)
- Using web scraping to predict unforseen economic events
- Deploying an auto-retraining pipeline based on new sales data

---
