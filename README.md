# Sales Forecasting Using Temporal Fusion Transformer (TFT)

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

- **Gated Residual Networks (GRN)** for feature processing
- **Variable Selection Networks** for dynamic feature selection
- **LSTM Encoder-Decoder** for capturing temporal relationships
- **Multi-Head Attention** for long-term dependency learning
- **Static Covariates Encoder** for embedding constant features

### Model Diagram
Here’s a high-level architecture overview:

```plaintext
Input Features (Static + Time-Varying)
            ↓
    Variable Selection Networks
            ↓
    Gated Residual Networks (GRN)
            ↓
        LSTM Encoder
            ↓
        LSTM Decoder
            ↓
    Multi-Head Attention Mechanism
            ↓
    Prediction Layer (Sales Forecast)
```

*Note: A detailed visualization of TFT is available in the [original TFT paper](https://arxiv.org/abs/1912.09363).*

---

## Example Prediction

Here’s an example of how the model predicts sales:

| Quarter        | Actual Sales | Predicted Sales |
|----------------|--------------|-----------------|
| Q1 2023        |  1,250,000   |   1,240,500     |
| Q2 2023        |  1,300,000   |   1,295,200     |
| Q3 2023        |  1,400,000   |   1,390,800     |
| Q4 2023        |  1,450,000   |   1,445,300     |

The model achieves high accuracy in predicting quarterly trends based on previous distributor sales records.

---

## Future Improvements
- Hyperparameter tuning for even better prediction accuracy
- Incorporating additional external covariates (e.g., economic indicators)
- Deploying an auto-retraining pipeline based on new sales data
