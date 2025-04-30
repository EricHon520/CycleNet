import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression  # 新增 LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add LightGBM
import lightgbm as lgbm
# 移除 XGBoost
# import xgboost as xgb
import time  # 用於追蹤訓練進度
import random  # 新增用於設定隨機種子

# 設定全局的隨機種子，確保結果可復現
def set_seed(seed):
    """設置所有可能的隨機種子以確保實驗可復現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 設定Python內建的hash隨機性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 設定NumPy的隨機性
    np.random.seed(seed)
    # 如果使用CUDA，則確保CUDA操作的確定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"設定全局隨機種子為: {seed}")


def load_weather_data(root_path='./dataset/', data_path='weather.csv', 
                   seq_len=96, pred_len=96, features='M'):
    """
    Load Weather dataset and prepare for machine learning models
    """
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    
    # Set data boundaries (consistent with the original code, using the same train/validation/test split)
    # Weather data boundaries setup (70%/20%/10%)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    
    # Select feature columns
    if features == 'M' or features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    else:
        # If single-variable prediction is needed, use WetBulbCelsius as the default target variable
        target = 'OT'
        df_data = df_raw[[target]]
    
    # Standardize data
    scaler = StandardScaler()
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)
    
    # Extract timestamps
    df_stamp = df_raw[['date']][border1s[0]:border2s[2]]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(columns=['date']).values
    
    # Return training, validation and test sets
    train_x = data[border1s[0]:border2s[0]-pred_len, :]
    train_y = data[border1s[0]+seq_len:border2s[0], :]
    
    val_x = data[border1s[1]:border2s[1]-pred_len, :]
    val_y = data[border1s[1]+seq_len:border2s[1], :]
    
    test_x = data[border1s[2]:border2s[2]-pred_len, :]
    test_y = data[border1s[2]+seq_len:border2s[2], :]
    
    # To adapt for machine learning models, we need to convert sequences to features
    train_x_reshaped = []
    val_x_reshaped = []
    test_x_reshaped = []
    
    # For each prediction point, extract historical data of length seq_len as features
    for i in range(len(train_x) - seq_len + 1):
        train_x_reshaped.append(train_x[i:i+seq_len].flatten())
    for i in range(len(val_x) - seq_len + 1):
        val_x_reshaped.append(val_x[i:i+seq_len].flatten())
    for i in range(len(test_x) - seq_len + 1):
        test_x_reshaped.append(test_x[i:i+seq_len].flatten())
    
    train_x_reshaped = np.array(train_x_reshaped)
    val_x_reshaped = np.array(val_x_reshaped)
    test_x_reshaped = np.array(test_x_reshaped)
    
    # To predict a sequence of length pred_len, prepare the label data accordingly
    # The y data should be the values of the next pred_len time steps
    train_y_reshaped = []
    val_y_reshaped = []
    test_y_reshaped = []
    
    for i in range(len(train_y) - pred_len + 1):
        train_y_reshaped.append(train_y[i:i+pred_len, :])
    for i in range(len(val_y) - pred_len + 1):
        val_y_reshaped.append(val_y[i:i+pred_len, :])
    for i in range(len(test_y) - pred_len + 1):
        test_y_reshaped.append(test_y[i:i+pred_len, :])
    
    train_y_reshaped = np.array(train_y_reshaped)
    val_y_reshaped = np.array(val_y_reshaped)
    test_y_reshaped = np.array(test_y_reshaped)
    
    # Adjust data shape for easier processing
    train_y_reshaped = train_y_reshaped.reshape(train_y_reshaped.shape[0], -1)
    val_y_reshaped = val_y_reshaped.reshape(val_y_reshaped.shape[0], -1)
    test_y_reshaped = test_y_reshaped.reshape(test_y_reshaped.shape[0], -1)
    
    return (train_x_reshaped, train_y_reshaped, 
            val_x_reshaped, val_y_reshaped, 
            test_x_reshaped, test_y_reshaped, 
            scaler)

def evaluate_metrics(pred, true):
    """
    Calculate evaluation metrics: MAE, MSE, RMSE
    """
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def evaluate_metrics_extended(pred, true):
    """
    Calculate the same evaluation metrics as in exp_main.py
    """
    # Ensure consistent shapes - handle multivariate cases
    pred = np.array(pred)
    true = np.array(true)
    
    # Basic metrics
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    # Advanced metrics
    # Prevent division by zero
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))
    mspe = np.mean(((true[mask] - pred[mask]) / true[mask]) ** 2)
    
    # Relative squared error
    mean_true = np.mean(true)
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - mean_true) ** 2)
    rse = numerator / denominator if denominator != 0 else np.inf
    
    # Correlation coefficient
    # Reshape to 2D arrays to calculate correlation
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    corr = np.corrcoef(pred_flat, true_flat)[0, 1] if len(pred_flat) > 1 else 0
    
    return mae, mse, rmse, mape, mspe, rse, corr

def train_and_evaluate_lgbm(train_x, train_y, val_x, val_y, test_x, test_y, 
                           pred_len, features_per_step, n_estimators=100, 
                           learning_rate=0.1, max_depth=-1, show_progress=True):
    """
    Train and evaluate LightGBM models
    """
    print(f"Starting LightGBM model training (n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth})...")
    
    # Get total number of variables
    n_variables = train_y.shape[1] // pred_len
    total_models = pred_len * n_variables
    models = []
    
    # Create a LightGBM model for each output time step and variable
    for i in range(pred_len):
        for j in range(n_variables):
            if show_progress:
                print(f"Training model {len(models) + 1}/{total_models}...")
                
            # Extract target value for current time step and feature
            target_idx = i * n_variables + j
            current_y = train_y[:, target_idx]
            
            # Create and train LightGBM model
            model = lgbm.LGBMRegressor(n_estimators=n_estimators, 
                                       learning_rate=learning_rate,
                                       max_depth=max_depth, 
                                       random_state=42)
            model.fit(train_x, current_y)
            models.append(model)
    
    # Predict on validation set
    val_pred = np.zeros((val_x.shape[0], val_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            val_pred[:, target_idx] = models[target_idx].predict(val_x)
    
    # Predict on test set
    test_pred = np.zeros((test_x.shape[0], test_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            test_pred[:, target_idx] = models[target_idx].predict(test_x)
    
    # Calculate evaluation metrics
    val_mae, val_mse, val_rmse = evaluate_metrics(val_pred, val_y)
    test_mae, test_mse, test_rmse = evaluate_metrics(test_pred, test_y)
    
    print("LightGBM model evaluation results:")
    print(f"Validation set - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    print(f"Test set - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    return models, val_pred, test_pred, (val_mae, val_mse, val_rmse), (test_mae, test_mse, test_rmse)

def train_and_evaluate_rf(train_x, train_y, val_x, val_y, test_x, test_y, 
                          pred_len, features_per_step, n_estimators=100, 
                          max_depth=None, show_progress=True):
    """
    Train and evaluate Random Forest models
    """
    print(f"Starting Random Forest model training (n_estimators={n_estimators}, max_depth={max_depth})...")
    
    # Since RandomForest is a single-output model, we need to train a separate model for each output time step and feature
    models = []
    
    # Get total number of variables
    n_variables = train_y.shape[1] // pred_len
    
    total_models = pred_len * n_variables
    
    # Create a RandomForest model for each output time step and variable
    for i in range(pred_len):
        for j in range(n_variables):
            if show_progress:
                print(f"Training model {len(models) + 1}/{total_models}...")
                
            # Extract target value for current time step and feature
            target_idx = i * n_variables + j
            current_y = train_y[:, target_idx]
            
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(train_x, current_y)
            models.append(model)
    
    # Predict on validation set
    val_pred = np.zeros((val_x.shape[0], val_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            val_pred[:, target_idx] = models[target_idx].predict(val_x)
    
    # Predict on test set
    test_pred = np.zeros((test_x.shape[0], test_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            test_pred[:, target_idx] = models[target_idx].predict(test_x)
    
    # Calculate evaluation metrics
    val_mae, val_mse, val_rmse = evaluate_metrics(val_pred, val_y)
    test_mae, test_mse, test_rmse = evaluate_metrics(test_pred, test_y)
    
    print("Random Forest model evaluation results:")
    print(f"Validation set - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    print(f"Test set - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    return models, val_pred, test_pred, (val_mae, val_mse, val_rmse), (test_mae, test_mse, test_rmse)

def train_and_evaluate_ridge(train_x, train_y, val_x, val_y, test_x, test_y, 
                           pred_len, features_per_step, alpha=1.0, show_progress=True, random_seed=42):
    """
    Train and evaluate Ridge linear models (much faster than tree-based models)
    """
    print(f"Starting Ridge regression model training (alpha={alpha})...")
    start_time = time.time()
    
    # Get total number of variables
    n_variables = train_y.shape[1] // pred_len
    total_models = pred_len * n_variables
    
    print(f"Need to train {total_models} Ridge linear sub-models")
    models = []
    completed = 0
    
    # Create a Ridge model for each output time step and variable
    for i in range(pred_len):
        for j in range(n_variables):
            # Extract target value for current time step and feature
            target_idx = i * n_variables + j
            current_y = train_y[:, target_idx]
            
            # Create and train Ridge model
            model = Ridge(alpha=alpha, random_state=random_seed)
            model.fit(train_x, current_y)
            models.append(model)
            
            completed += 1
            if show_progress and completed % 50 == 0:  # 每 50 個模型顯示一次進度
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_models - completed) if completed > 0 else 0
                print(f"Train Ridge model: {completed}/{total_models} [timestep {i+1}/{pred_len}, number of variables {j+1}/{n_variables}]")
                print(f"Elapsed time: {elapsed:.1f} seconds, estimated remaining time: {eta:.1f} seconds")
    
    # Predict on validation set
    val_pred = np.zeros((val_x.shape[0], val_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            val_pred[:, target_idx] = models[target_idx].predict(val_x)
    
    # Predict on test set
    test_pred = np.zeros((test_x.shape[0], test_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            test_pred[:, target_idx] = models[target_idx].predict(test_x)
    
    # Calculate evaluation metrics
    val_mae, val_mse, val_rmse = evaluate_metrics(val_pred, val_y)
    test_mae, test_mse, test_rmse = evaluate_metrics(test_pred, test_y)
    
    total_time = time.time() - start_time
    print(f"Ridge model training completed! Total time: {total_time:.2f} seconds")
    print("Ridge model evaluation results:")
    print(f"Validation set - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    print(f"Test set - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    return models, val_pred, test_pred, (val_mae, val_mse, val_rmse), (test_mae, test_mse, test_rmse)

def train_and_evaluate_linear(train_x, train_y, val_x, val_y, test_x, test_y, 
                             pred_len, features_per_step, show_progress=True, random_seed=42):
    """
    訓練和評估線性回歸模型 (比 XGBoost 快很多倍)
    """
    print(f"Start linear regression model training...")
    start_time = time.time()

    # 獲取變量總數
    n_variables = train_y.shape[1] // pred_len
    total_models = pred_len * n_variables
    
    print(f"Need to train {total_models} linear regression sub-models")
    models = []
    completed = 0
    
    # 為每個輸出時間步長和變量創建一個線性回歸模型
    for i in range(pred_len):
        for j in range(n_variables):
            # 提取當前時間步長和特徵的目標值
            target_idx = i * n_variables + j
            current_y = train_y[:, target_idx]
            
            # 創建並訓練線性回歸模型
            model = LinearRegression()
            model.fit(train_x, current_y)
            models.append(model)
            
            completed += 1
            if show_progress and completed % 100 == 0:  # 每 100 個模型顯示一次進度
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total_models - completed) if completed > 0 else 0
                print(f"Train linear model: {completed}/{total_models} [timestep {i+1}/{pred_len}, number of variables {j+1}/{n_variables}]")
                print(f"Elapsed time: {elapsed:.1f} seconds, estimated remaining time: {eta:.1f} seconds")
    
    # 在驗證集上預測
    val_pred = np.zeros((val_x.shape[0], val_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            val_pred[:, target_idx] = models[target_idx].predict(val_x)
    
    # 在測試集上預測
    test_pred = np.zeros((test_x.shape[0], test_y.shape[1]))
    for i in range(pred_len):
        for j in range(n_variables):
            target_idx = i * n_variables + j
            test_pred[:, target_idx] = models[target_idx].predict(test_x)
    
    # 計算評估指標
    val_mae, val_mse, val_rmse = evaluate_metrics(val_pred, val_y)
    test_mae, test_mse, test_rmse = evaluate_metrics(test_pred, test_y)
    
    total_time = time.time() - start_time
    print(f"Linear regression model training is complete! Total time: {total_time:.2f} seconds")
    print("Linear regression model evaluation results:")
    print(f"Validation set - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    print(f"Test set - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    return models, val_pred, test_pred, (val_mae, val_mse, val_rmse), (test_mae, test_mse, test_rmse)

def plot_predictions(true_vals, pred_vals, step_idx=0, var_idx=0, n_points=100, title="Predictions vs True Values"):
    """
    Plot comparison of predictions and true values
    """
    n_variables = true_vals.shape[1] // pred_len
    idx = step_idx * n_variables + var_idx
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_vals[:n_points, idx], label='True')
    plt.plot(pred_vals[:n_points, idx], label='Predicted')
    plt.legend()
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

def visualize_predictions(pred, true, folder_path, seq_len, pred_len, samples=5):
    """
    Plot comparison charts of predictions and true values
    """
    # Select number of samples
    sample_count = min(samples, pred.shape[0])
    
    # Plot charts for each selected sample
    for i in range(sample_count):
        # Get single time series sample
        # For multivariate data, we select the first variable for visualization
        if len(pred.shape) > 2:
            target_var = 0  # Can be changed according to needs
            sample_pred = pred[i, :, target_var]
            sample_true = true[i, :, target_var]
        else:
            sample_pred = pred[i]
            sample_true = true[i]
        
        plt.figure(figsize=(12, 6))
        
        # Create sequence index
        t = np.arange(0, len(sample_true))
        
        # Plot predicted and true values
        plt.plot(t, sample_true, label='True', color='blue', linestyle='-')
        plt.plot(t, sample_pred, label='Predicted', color='red', linestyle='--')
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Sample {i+1}: Prediction vs True Values')
        plt.grid(True)
        
        # Save image
        plt.savefig(os.path.join(folder_path, f'sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    '''
    # If there are multiple variables, plot average performance chart
    if len(pred.shape) > 2 and pred.shape[2] > 1:
        plot_multivariate_performance(pred, true, folder_path)
    '''

def plot_multivariate_performance(pred, true, folder_path):
    """
    Plot performance analysis charts for multivariate time series
    
    Parameters:
    - pred: Predicted values [samples, time_steps, variables]
    - true: True values [samples, time_steps, variables]
    - folder_path: Directory to save charts
    """
    if len(pred.shape) < 3 or len(true.shape) < 3:
        print("Multivariate data is required to plot multivariate performance charts")
        return
        
    n_variables = pred.shape[2]
    
    # Calculate performance metrics for each variable
    var_metrics = []
    for i in range(n_variables):
        mae, mse, rmse, mape, mspe, rse, corr = evaluate_metrics_extended(
            pred[:, :, i], true[:, :, i]
        )
        var_metrics.append({
            'mae': mae, 
            'rmse': rmse, 
            'mape': mape,
            'corr': corr
        })
    
    # Plot comparison charts for different variables' performance
    metrics_to_plot = ['mae', 'rmse', 'mape', 'corr']
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        metric_values = [metrics[metric_name] for metrics in var_metrics]
        
        plt.bar(range(n_variables), metric_values, color='skyblue')
        plt.xlabel('Variable Index')
        plt.ylabel(metric_name.upper())
        plt.title(f'{metric_name.upper()} Metric for Each Variable')
        plt.xticks(range(n_variables))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save chart
        plt.savefig(os.path.join(folder_path, f'multivariate_{metric_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot heatmap showing prediction accuracy for different variables
    plt.figure(figsize=(10, 8))
    metrics_array = np.array([[m['mae'], m['rmse'], m['mape'], m['corr']] for m in var_metrics])
    
    # Handle potential infinity or NaN values
    metrics_array = np.nan_to_num(metrics_array, nan=0, posinf=1, neginf=0)
    
    plt.imshow(metrics_array, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='Metric Value')
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Variable Index')
    plt.title('Multivariate Prediction Performance Heatmap')
    plt.xticks(range(4), ['MAE', 'RMSE', 'MAPE', 'CORR'])
    plt.yticks(range(n_variables))
    
    plt.savefig(os.path.join(folder_path, 'multivariate_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    metrics_df = pd.DataFrame(var_metrics)
    metrics_df.index = [f'Variable_{i}' for i in range(n_variables)]
    metrics_df.to_csv(os.path.join(folder_path, 'multivariate_metrics.csv'))

def save_ml_results(pred, true, setting, model_name, seq_len, pred_len):
    """
    Save results in a format compatible with deep learning models
    """
    # Create results directory
    folder_path = f'./results/{setting}_{model_name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Calculate evaluation metrics
    mae, mse, rmse, mape, mspe, rse, corr = evaluate_metrics_extended(pred, true)
    
    # Print and save main metrics
    print(f'{model_name} - mse:{mse:.4f}, mae:{mae:.4f}')
    with open("result.txt", 'a') as f:
        f.write(f"{setting}_{model_name}  \n")
        f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}, rse:{rse:.4f}, corr:{corr:.4f}')
        f.write('\n\n')
    
    # Save evaluation metrics
    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
    np.save(folder_path + 'pred.npy', pred)
    np.save(folder_path + 'true.npy', true)
    
    # Visualize some samples
    visualize_predictions(pred, true, folder_path, seq_len, pred_len)
    
    return mae, mse, rmse, mape, mspe, rse, corr

if __name__ == "__main__":
    # Parameter settings
    seq_len = 96
    # 使用與 weather.sh 相同的預測長度設定
    pred_len_list = [96]
    # 使用與 weather.sh 相同的隨機種子設定
    random_seed_list = [2024]
    features = 'M'  # Use multivariate prediction
    
    # 計算總訓練模型數量
    total_pred_len = sum(pred_len_list)
    features_per_step = 21  # Weather dataset has 21 variables
    total_models = total_pred_len * features_per_step * len(random_seed_list) * 1  # 2 types of models
    
    print(f"=" * 80)
    print(f"A total of {total_models} models need to be trained")
    print(f"Contains {len(pred_len_list)} prediction lengths × {features_per_step} variables × 2 model types × {len(random_seed_list)} random seeds")
    print(f"=" * 80)
    
    main_start_time = time.time()
    completed_models = 0
    
    # 針對每個預測長度運行模型
    for pred_len in pred_len_list:
        print(f"\n{'-'*50}")
        print(f"Running for prediction length: {pred_len}")
        print(f"{'-'*50}")
        
        # Load weather data
        print(f"Loading Weather data, sequence length={seq_len}, prediction length={pred_len}")
        train_x, train_y, val_x, val_y, test_x, test_y, scaler = load_weather_data(
            seq_len=seq_len, pred_len=pred_len, features=features
        )
        
        # Weather dataset has 21 variables
        features_per_step = 21  # Number of features in weather.csv
        
        print(f"Training data shape: {train_x.shape}, {train_y.shape}")
        print(f"Validation data shape: {val_x.shape}, {val_y.shape}")
        print(f"Test data shape: {test_x.shape}, {test_y.shape}")
        
        # 存儲每個隨機種子的結果
        linear_results = []
        ridge_results = []
        
        # 針對每個隨機種子運行模型
        for random_seed in random_seed_list:
            print(f"\nProcess random seed: {random_seed}, prediction length: {pred_len}")
            print(f"Total progress: {completed_models}/{total_models} completed")
            
            # Train and evaluate Ridge model (linear model - much faster)
            ridge_models, ridge_val_pred, ridge_test_pred, ridge_val_metrics, ridge_test_metrics = train_and_evaluate_ridge(
                train_x, train_y, val_x, val_y, test_x, test_y, 
                pred_len=pred_len, features_per_step=features_per_step,
                alpha=1.0, show_progress=True, random_seed=random_seed
            )

            completed_models += pred_len * features_per_step
            print(f"Total progress: {completed_models}/{total_models} completed")
            
            # Train and evaluate Linear Regression model
            linear_models, linear_val_pred, linear_test_pred, linear_val_metrics, linear_test_metrics = train_and_evaluate_linear(
                train_x, train_y, val_x, val_y, test_x, test_y, 
                pred_len=pred_len, features_per_step=features_per_step,
                show_progress=True, random_seed=random_seed
            )
            
            completed_models += pred_len * features_per_step
            elapsed = time.time() - main_start_time
            eta = (elapsed / completed_models) * (total_models - completed_models) if completed_models > 0 else 0
            print(f"Total progress: {completed_models}/{total_models} completed")
            print(f"Total time: {elapsed/60:.1f} minutes, estimated remaining time: {eta/60:.1f} minutes")
            
            # 儲存每個隨機種子的結果
            linear_results.append({
                'seed': random_seed,
                'val_metrics': linear_val_metrics,
                'test_metrics': linear_test_metrics,
                'test_pred': linear_test_pred
            })
            ridge_results.append({
                'seed': random_seed,
                'val_metrics': ridge_val_metrics,
                'test_metrics': ridge_test_metrics,
                'test_pred': ridge_test_pred
            })
            
            # 保存帶有隨機種子信息的結果
            setting = f"weather_{seq_len}_{pred_len}_seed{random_seed}"
            save_ml_results(linear_test_pred, test_y, setting, "LinearRegression", seq_len, pred_len)
            save_ml_results(ridge_test_pred, test_y, setting, "Ridge", seq_len, pred_len)
        
        # 計算並打印平均結果
        print(f"\n{'-'*50}")
        print(f"Average results of prediction length {pred_len} under {len(random_seed_list)} seeds:")
        
        # Linear Regression 平均結果
        linear_avg_mae = np.mean([res['test_metrics'][0] for res in linear_results])
        linear_avg_rmse = np.mean([res['test_metrics'][2] for res in linear_results])
        
        # Ridge 平均結果
        ridge_avg_mae = np.mean([res['test_metrics'][0] for res in ridge_results])
        ridge_avg_rmse = np.mean([res['test_metrics'][2] for res in ridge_results])
        
        print("Model\t\tAverage MAE\t\tAverage RMSE")
        print(f"Linear Regression\t{linear_avg_mae:.4f}\t\t{linear_avg_rmse:.4f}")
        print(f"Ridge Regression\t{ridge_avg_mae:.4f}\t\t{ridge_avg_rmse:.4f}")
        
        # 為平均結果繪圖
        best_linear_idx = np.argmin([res['test_metrics'][0] for res in linear_results])
        worst_linear_idx = np.argmax([res['test_metrics'][0] for res in linear_results])
        best_ridge_idx = np.argmin([res['test_metrics'][0] for res in ridge_results])
        worst_ridge_idx = np.argmax([res['test_metrics'][0] for res in ridge_results])
        
        plot_predictions(test_y, linear_results[best_linear_idx]['test_pred'], 
                         title=f"Weather Linear Regression (pred_len={pred_len}) - Best Results")
        plot_predictions(test_y, linear_results[worst_linear_idx]['test_pred'],
                         title=f"Weather Linear Regression (pred_len={pred_len}) - Worst Results")
        plot_predictions(test_y, ridge_results[best_ridge_idx]['test_pred'], 
                         title=f"Weather Ridge (pred_len={pred_len}) - Best Results")
        plot_predictions(test_y, ridge_results[worst_ridge_idx]['test_pred'],
                         title=f"Weather Ridge (pred_len={pred_len}) - Worst Results")
    
    total_time = time.time() - main_start_time
    print(f"\n{'-'*50}")
    print(f"All model training is complete!")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Average training time per model: {total_time/total_models:.4f} seconds")
    print(f"{'-'*50}")