import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import Dataset, DataLoader

# Load the data
train_data = pd.read_csv('./data/train_data.csv')
val_data = pd.read_csv('./data/val_data.csv')

# Preprocessing function
def preprocess_data(data):
    # Convert categorical variables to one-hot encoding
    categorical_features = ['distributor_id', 'industry', 'sku', 'category', 'movement_category']
    numerical_features = ['sales', 'avg_quarterly_sales', 'total_quarter_sales', 'prev_quarter_sales', 
                         'is_diwali', 'is_ganesh_chaturthi', 'is_gudi_padwa', 'is_eid', 
                         'is_akshay_tritiya', 'is_dussehra_navratri', 'is_onam', 'is_christmas', 'time_idx']
    
    # One-hot encoding for categorical features
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(data[categorical_features])
    
    # Scaling numerical features
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(data[numerical_features])
    
    # Combine processed features
    processed_data = np.hstack((encoded_categorical, scaled_numerical))
    
    return processed_data

# Preprocess train and validation data
X_train = preprocess_data(train_data)
y_train = train_data['sales'].values  # Target variable
X_val = preprocess_data(val_data)
y_val = val_data['sales'].values  # Target variable