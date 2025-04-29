import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 添加 LightGBM
import lightgbm as lgbm


def load_weather_data(root_path='./dataset/', data_path='weather.csv', 
                   seq_len=96, pred_len=96, features='M'):
    """
    加載Weather數據集，並準備用於機器學習模型
    """
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    
    # 設置數據邊界（與原始程式碼一致，使用相同的訓練/驗證/測試劃分）
    # Weather數據的邊界設置 (70%/20%/10%)
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    
    # 選擇特徵列
    if features == 'M' or features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    else:
        # 如果需要單變量預測，默認使用WetBulbCelsuis作為目標變量
        target = 'OT'
        df_data = df_raw[[target]]
    
    # 標準化數據
    scaler = StandardScaler()
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)
    
    # 提取時間戳
    df_stamp = df_raw[['date']][border1s[0]:border2s[2]]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(columns=['date']).values
    
    # 返回訓練集、驗證集和測試集
    train_x = data[border1s[0]:border2s[0]-pred_len, :]
    train_y = data[border1s[0]+seq_len:border2s[0], :]
    
    val_x = data[border1s[1]:border2s[1]-pred_len, :]
    val_y = data[border1s[1]+seq_len:border2s[1], :]
    
    test_x = data[border1s[2]:border2s[2]-pred_len, :]
    test_y = data[border1s[2]+seq_len:border2s[2], :]
    
    # 為了適用於機器學習模型，我們需要將序列轉換為特徵
    train_x_reshaped = []
    val_x_reshaped = []
    test_x_reshaped = []
    
    # 為每個預測點，提取seq_len長度的歷史數據作為特徵
    for i in range(len(train_x) - seq_len + 1):
        train_x_reshaped.append(train_x[i:i+seq_len].flatten())
    for i in range(len(val_x) - seq_len + 1):
        val_x_reshaped.append(val_x[i:i+seq_len].flatten())
    for i in range(len(test_x) - seq_len + 1):
        test_x_reshaped.append(test_x[i:i+seq_len].flatten())
    
    train_x_reshaped = np.array(train_x_reshaped)
    val_x_reshaped = np.array(val_x_reshaped)
    test_x_reshaped = np.array(test_x_reshaped)
    
    # 為了預測pred_len長度的序列，需要相應地準備標籤數據
    # 對應的y數據需要是未來pred_len個時間步的值
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
    
    # 調整數據形狀以方便後續處理
    train_y_reshaped = train_y_reshaped.reshape(train_y_reshaped.shape[0], -1)
    val_y_reshaped = val_y_reshaped.reshape(val_y_reshaped.shape[0], -1)
    test_y_reshaped = test_y_reshaped.reshape(test_y_reshaped.shape[0], -1)
    
    return (train_x_reshaped, train_y_reshaped, 
            val_x_reshaped, val_y_reshaped, 
            test_x_reshaped, test_y_reshaped, 
            scaler)

def evaluate_metrics(pred, true):
    """
    計算評估指標：MAE, MSE, RMSE
    """
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def evaluate_metrics_extended(pred, true):
    """
    計算與exp_main.py完全相同的評估指標
    """
    # 確保形狀一致 - 處理多變量情況
    pred = np.array(pred)
    true = np.array(true)
    
    # 基本指標
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    # 進階指標
    # 防止除以零
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))
    mspe = np.mean(((true[mask] - pred[mask]) / true[mask]) ** 2)
    
    # 相對平方誤差
    mean_true = np.mean(true)
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - mean_true) ** 2)
    rse = numerator / denominator if denominator != 0 else np.inf
    
    # 相關係數
    # 重塑為二維數組計算相關性
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    corr = np.corrcoef(pred_flat, true_flat)[0, 1] if len(pred_flat) > 1 else 0
    
    return mae, mse, rmse, mape, mspe, rse, corr

def train_and_evaluate_lgbm(train_x, train_y, val_x, val_y, test_x, test_y, 
                           pred_len, features_per_step, n_estimators=100, 
                           learning_rate=0.1, max_depth=-1, show_progress=True):
    """
    訓練與評估LightGBM模型
    """
    print(f"開始訓練LightGBM模型 (n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth})...")
    
    # 獲取總變量數
    n_variables = train_y.shape[1] // pred_len
    total_models = pred_len * n_variables
    models = []
    
    # 為每個輸出時間步和變量創建一個LightGBM模型
    for i in range(pred_len):
        for j in range(n_variables):
            if show_progress:
                print(f"訓練模型 {len(models) + 1}/{total_models}...")
                
            # 提取當前時間步和特徵的目標值
            target_idx = i * n_variables + j
            current_y = train_y[:, target_idx]
            
            # 創建並訓練LightGBM模型
            model = lgbm.LGBMRegressor(n_estimators=n_estimators, 
                                       learning_rate=learning_rate,
                                       max_depth=max_depth, 
                                       random_state=42)
            model.fit(train_x, current_y)
            models.append(model)
    
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
    
    print("LightGBM模型評估結果:")
    print(f"驗證集 - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    print(f"測試集 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    return models, val_pred, test_pred, (val_mae, val_mse, val_rmse), (test_mae, test_mse, test_rmse)

def train_and_evaluate_rf(train_x, train_y, val_x, val_y, test_x, test_y, 
                          pred_len, features_per_step, n_estimators=100, 
                          max_depth=None, show_progress=True):
    """
    訓練與評估隨機森林模型
    """
    print(f"開始訓練隨機森林模型 (n_estimators={n_estimators}, max_depth={max_depth})...")
    
    # 由於RandomForest是單輸出模型，我們需要為每個輸出時間步的每個特徵訓練一個單獨的模型
    models = []
    
    # 獲取總變量數
    n_variables = train_y.shape[1] // pred_len
    
    total_models = pred_len * n_variables
    
    # 為每個輸出時間步和變量創建一個RandomForest模型
    for i in range(pred_len):
        for j in range(n_variables):
            if show_progress:
                print(f"訓練模型 {len(models) + 1}/{total_models}...")
                
            # 提取當前時間步和特徵的目標值
            target_idx = i * n_variables + j
            current_y = train_y[:, target_idx]
            
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(train_x, current_y)
            models.append(model)
    
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
    
    print("隨機森林模型評估結果:")
    print(f"驗證集 - MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    print(f"測試集 - MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    return models, val_pred, test_pred, (val_mae, val_mse, val_rmse), (test_mae, test_mse, test_rmse)

def plot_predictions(true_vals, pred_vals, step_idx=0, var_idx=0, n_points=100, title="Predictions vs True Values"):
    """
    繪製預測結果與真實值的對比圖
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
    繪製預測結果與真實值的比較圖
    """
    # 選擇樣本數
    sample_count = min(samples, pred.shape[0])
    
    # 對每個選定樣本繪製圖形
    for i in range(sample_count):
        # 取單個時間序列樣本
        # 對於多變量，我們選擇第一個變量進行可視化
        if len(pred.shape) > 2:
            target_var = 0  # 可以根據需要更改
            sample_pred = pred[i, :, target_var]
            sample_true = true[i, :, target_var]
        else:
            sample_pred = pred[i]
            sample_true = true[i]
        
        plt.figure(figsize=(12, 6))
        
        # 創建序列索引
        t = np.arange(0, len(sample_true))
        
        # 繪製預測值和真實值
        plt.plot(t, sample_true, label='True', color='blue', linestyle='-')
        plt.plot(t, sample_pred, label='Predicted', color='red', linestyle='--')
        
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Sample {i+1}: Prediction vs True Values')
        plt.grid(True)
        
        # 保存圖片
        plt.savefig(os.path.join(folder_path, f'sample_{i+1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 如果有多個變量，繪製平均性能圖
    if len(pred.shape) > 2 and pred.shape[2] > 1:
        plot_multivariate_performance(pred, true, folder_path)

def plot_multivariate_performance(pred, true, folder_path):
    """
    為多變量時間序列繪製性能分析圖
    
    參數:
    - pred: 預測值 [samples, time_steps, variables]
    - true: 真實值 [samples, time_steps, variables]
    - folder_path: 保存圖表的目錄
    """
    if len(pred.shape) < 3 or len(true.shape) < 3:
        print("需要多變量數據才能繪製多變量性能圖")
        return
        
    n_variables = pred.shape[2]
    
    # 計算每個變量的性能指標
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
    
    # 繪製不同變量的性能對比圖
    metrics_to_plot = ['mae', 'rmse', 'mape', 'corr']
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(12, 6))
        metric_values = [metrics[metric_name] for metrics in var_metrics]
        
        plt.bar(range(n_variables), metric_values, color='skyblue')
        plt.xlabel('變量索引')
        plt.ylabel(metric_name.upper())
        plt.title(f'各變量的{metric_name.upper()}指標')
        plt.xticks(range(n_variables))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 保存圖表
        plt.savefig(os.path.join(folder_path, f'multivariate_{metric_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 繪製熱力圖，展示不同變量的預測準確度
    plt.figure(figsize=(10, 8))
    metrics_array = np.array([[m['mae'], m['rmse'], m['mape'], m['corr']] for m in var_metrics])
    
    # 處理可能的無限值或NaN
    metrics_array = np.nan_to_num(metrics_array, nan=0, posinf=1, neginf=0)
    
    plt.imshow(metrics_array, cmap='coolwarm', aspect='auto')
    plt.colorbar(label='指標值')
    plt.xlabel('評估指標')
    plt.ylabel('變量索引')
    plt.title('多變量預測性能熱力圖')
    plt.xticks(range(4), ['MAE', 'RMSE', 'MAPE', 'CORR'])
    plt.yticks(range(n_variables))
    
    plt.savefig(os.path.join(folder_path, 'multivariate_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存數值結果
    metrics_df = pd.DataFrame(var_metrics)
    metrics_df.index = [f'變量_{i}' for i in range(n_variables)]
    metrics_df.to_csv(os.path.join(folder_path, 'multivariate_metrics.csv'))

def save_ml_results(pred, true, setting, model_name, seq_len, pred_len):
    """
    保存與深度學習模型兼容的結果格式
    """
    # 創建結果目錄
    folder_path = f'./results/{setting}_{model_name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 計算評估指標
    mae, mse, rmse, mape, mspe, rse, corr = evaluate_metrics_extended(pred, true)
    
    # 打印和保存主要指標
    print(f'{model_name} - mse:{mse:.4f}, mae:{mae:.4f}')
    with open("result.txt", 'a') as f:
        f.write(f"{setting}_{model_name}  \n")
        f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}, rse:{rse:.4f}, corr:{corr:.4f}')
        f.write('\n\n')
    
    # 保存評估指標
    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
    np.save(folder_path + 'pred.npy', pred)
    np.save(folder_path + 'true.npy', true)
    
    # 視覺化部分樣本
    visualize_predictions(pred, true, folder_path, seq_len, pred_len)
    
    return mae, mse, rmse, mape, mspe, rse, corr

if __name__ == "__main__":
    # 參數設置
    seq_len = 96
    pred_len = 96
    features = 'M'  # 使用多變量預測
    
    # 加載weather數據
    print(f"加載Weather數據，序列長度={seq_len}，預測長度={pred_len}")
    train_x, train_y, val_x, val_y, test_x, test_y, scaler = load_weather_data(
        seq_len=seq_len, pred_len=pred_len, features=features
    )
    
    # Weather數據集有21個變量
    features_per_step = 21  # weather.csv中的特徵數量
    
    print(f"訓練資料形狀: {train_x.shape}, {train_y.shape}")
    print(f"驗證資料形狀: {val_x.shape}, {val_y.shape}")
    print(f"測試資料形狀: {test_x.shape}, {test_y.shape}")
    
    # 訓練與評估LightGBM模型
    lgbm_models, lgbm_val_pred, lgbm_test_pred, lgbm_val_metrics, lgbm_test_metrics = train_and_evaluate_lgbm(
        train_x, train_y, val_x, val_y, test_x, test_y, 
        pred_len=pred_len, features_per_step=features_per_step,
        n_estimators=200, learning_rate=0.1, max_depth=-1  # -1表示不限制樹的深度
    )
    
    # 訓練與評估隨機森林模型
    rf_models, rf_val_pred, rf_test_pred, rf_val_metrics, rf_test_metrics = train_and_evaluate_rf(
        train_x, train_y, val_x, val_y, test_x, test_y, 
        pred_len=pred_len, features_per_step=features_per_step,
        n_estimators=200, max_depth=10
    )
    
    # 繪製預測結果
    plot_predictions(test_y, lgbm_test_pred, title="Weather LightGBM Predictions vs True Values")
    plot_predictions(test_y, rf_test_pred, title="Weather Random Forest Predictions vs True Values")
    
    # 保存結果
    save_ml_results(lgbm_test_pred, test_y, "weather", "LightGBM", seq_len, pred_len)
    save_ml_results(rf_test_pred, test_y, "weather", "RandomForest", seq_len, pred_len)
    
    # 輸出模型比較結果
    print("\n模型比較 (測試集結果):")
    print("模型\t\tMAE\t\tRMSE")
    print(f"LightGBM\t{lgbm_test_metrics[0]:.4f}\t\t{lgbm_test_metrics[2]:.4f}")
    print(f"RandomForest\t{rf_test_metrics[0]:.4f}\t\t{rf_test_metrics[2]:.4f}")