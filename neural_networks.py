import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')


def load_weather_data(root_path='./dataset/', data_path='weather.csv', 
                   seq_len=96, pred_len=96, features='M'):
    """
    Load Weather dataset and prepare for neural network models
    和ml_algorithms.py中的函數類似，但為神經網路模型準備張量數據
    """
    df_raw = pd.read_csv(os.path.join(root_path, data_path))
    
    # 設置數據邊界 (按照原始代碼, 使用相同的訓練/驗證/測試分割)
    # 天氣數據邊界設置 (70%/20%/10%)
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
        # 如果需要單變量預測, 使用OT作為預設目標變量
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
    
    # 為神經網路準備數據
    train_x, train_y = prepare_data_for_nn(data[border1s[0]:border2s[0]], seq_len, pred_len)
    val_x, val_y = prepare_data_for_nn(data[border1s[1]:border2s[1]], seq_len, pred_len)
    test_x, test_y = prepare_data_for_nn(data[border1s[2]:border2s[2]], seq_len, pred_len)
    
    # 轉換為PyTorch張量
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    val_x = torch.FloatTensor(val_x)
    val_y = torch.FloatTensor(val_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, scaler


def prepare_data_for_nn(data, seq_len, pred_len):
    """
    將數據準備為神經網路輸入的序列形式
    """
    x_data = []
    y_data = []
    
    # 創建輸入/輸出序列對
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:i+seq_len]
        y = data[i+seq_len:i+seq_len+pred_len]
        x_data.append(x)
        y_data.append(y)
    
    return np.array(x_data), np.array(y_data)


def evaluate_metrics(pred, true):
    """
    計算評估指標: MAE, MSE, RMSE, MAPE, MSPE, RSE, CORR
    """
    # 確保一致的形狀 - 處理多變量情況
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
    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    corr = np.corrcoef(pred_flat, true_flat)[0, 1] if len(pred_flat) > 1 else 0
    
    return mae, mse, rmse, mape, mspe, rse, corr


def save_nn_results(pred, true, setting, model_name, seq_len, pred_len):
    """
    保存結果，格式與深度學習模型兼容
    """
    # 創建結果目錄
    folder_path = f'./results/{setting}_{model_name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 計算評估指標
    mae, mse, rmse, mape, mspe, rse, corr = evaluate_metrics(pred, true)
    
    # 打印並保存主要指標
    print(f'{model_name} - mse:{mse:.4f}, mae:{mae:.4f}')
    with open("result.txt", 'a') as f:
        f.write(f"{setting}_{model_name}  \n")
        f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}, rse:{rse:.4f}, corr:{corr:.4f}')
        f.write('\n\n')
    
    # 保存評估指標
    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
    np.save(folder_path + 'pred.npy', pred)
    np.save(folder_path + 'true.npy', true)
    
    # 視覺化樣本
    visualize_predictions(pred, true, folder_path, seq_len, pred_len)
    
    return mae, mse, rmse, mape, mspe, rse, corr


def visualize_predictions(pred, true, folder_path, seq_len, pred_len, samples=5):
    """
    繪製預測值和真實值的比較圖
    """
    # 選擇樣本數量
    n_samples = min(samples, len(pred))
    
    # 取得變量數量
    if len(pred.shape) > 2:
        n_vars = pred.shape[2]
        # 使用第一個變量進行可視化
        target_var = 0
        
        # 為每個選擇的樣本繪製圖表
        for i in range(n_samples):
            plt.figure(figsize=(12, 6))
            plt.plot(true[i, :, target_var], label='真實值', color='blue', linestyle='-')
            plt.plot(pred[i, :, target_var], label='預測值', color='red', linestyle='--')
            plt.xlabel('時間步')
            plt.ylabel('值')
            plt.legend()
            plt.title(f'樣本 {i+1}: 預測 vs 真實值 (變量 {target_var})')
            plt.grid(True)
            plt.savefig(os.path.join(folder_path, f'sample_{i+1}_var_{target_var}.png'), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    else:
        # 單變量數據
        for i in range(n_samples):
            plt.figure(figsize=(12, 6))
            plt.plot(true[i], label='真實值', color='blue', linestyle='-')
            plt.plot(pred[i], label='預測值', color='red', linestyle='--')
            plt.xlabel('時間步')
            plt.ylabel('值')
            plt.legend()
            plt.title(f'樣本 {i+1}: 預測 vs 真實值')
            plt.grid(True)
            plt.savefig(os.path.join(folder_path, f'sample_{i+1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 繪製整體性能
    plt.figure(figsize=(10, 6))
    true_mean = np.mean(true, axis=0)
    pred_mean = np.mean(pred, axis=0)
    
    if len(true_mean.shape) > 1:
        true_mean = true_mean[:, target_var]
        pred_mean = pred_mean[:, target_var]
    
    plt.plot(true_mean, label='真實平均值', color='blue', linestyle='-')
    plt.plot(pred_mean, label='預測平均值', color='red', linestyle='--')
    plt.xlabel('時間步')
    plt.ylabel('平均值')
    plt.legend()
    plt.title('所有樣本的平均預測 vs 真實值')
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, 'average_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


# 定義多層感知器 (MLP) 模型
class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim, hidden_dims=[64, 128, 64]):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.flatten_dim = seq_len * input_dim
        
        # 定義網絡層
        layers = []
        input_size = self.flatten_dim
        
        # 添加隱藏層
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_dim
            
        # 輸出層
        self.output_layer = nn.Linear(input_size, pred_len * input_dim)
        
        # 建立模型
        self.hidden_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        # 展平輸入序列
        x = x.reshape(batch_size, -1)
        # 通過隱藏層
        x = self.hidden_layers(x)
        # 輸出層
        x = self.output_layer(x)
        # 重塑為序列形式
        return x.reshape(batch_size, self.pred_len, self.input_dim)


# 定義卷積神經網絡 (CNN) 模型
class CNNModel(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim, kernel_size=3):
        super(CNNModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        
        # 卷積層
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size, padding='same')
        self.conv3 = nn.Conv1d(64, 128, kernel_size, padding='same')
        
        # 池化層
        self.pool = nn.MaxPool1d(2)
        
        # 計算全連接層輸入大小
        self.flatten_size = 128 * (seq_len // 4)
        
        # 全連接層
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, pred_len * input_dim)
        
        # 激活函數和dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 調整維度順序為 [batch, channels, seq_len]
        x = x.permute(0, 2, 1)
        
        # 卷積塊 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # 卷積塊 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # 卷積塊 3
        x = self.relu(self.conv3(x))
        
        # 展平
        x = x.reshape(batch_size, -1)
        
        # 全連接層
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 重塑為預測序列 [batch, pred_len, input_dim]
        return x.reshape(batch_size, self.pred_len, self.input_dim)


# 定義循環神經網路 (RNN) 模型
class RNNModel(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim, hidden_dim=64, num_layers=2, rnn_type='lstm'):
        super(RNNModel, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 選擇RNN類型
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 輸出層
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 通過RNN層
        outputs, _ = self.rnn(x)
        
        # 只取最後一個時間步的輸出作為解碼器的初始輸入
        decoder_input = outputs[:, -1:, :]
        
        # 自回歸預測
        predictions = []
        current_input = decoder_input
        
        for _ in range(self.pred_len):
            # 通過一次時間步
            current_output, _ = self.rnn(current_input, None)
            # 應用輸出層得到預測
            current_pred = self.fc(current_output)
            # 添加到預測結果
            predictions.append(current_pred)
            # 更新當前輸入
            current_input = current_pred
        
        # 連接所有預測結果 [batch, pred_len, input_dim]
        predictions = torch.cat(predictions, dim=1)
        
        return predictions


def train_and_evaluate_model(model, model_name, train_x, train_y, val_x, val_y, test_x, test_y, 
                            batch_size=256, epochs=30, learning_rate=0.005, patience=5, device='cpu', random_seed=2024):
    """
    訓練和評估神經網絡模型
    """
    print(f"開始訓練 {model_name} 模型...")
    
    # 設置隨機種子以確保可重複性
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    
    # 將模型移至指定設備
    model = model.to(device)
    
    # 準備數據載入器
    train_dataset = TensorDataset(train_x.to(device), train_y.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_x.to(device), val_y.to(device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 損失函數和優化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 記錄最佳模型和驗證損失
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 訓練迴圈
    for epoch in range(epochs):
        # 訓練階段
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向傳播和優化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # 驗證階段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
        val_loss /= len(val_loader)
        
        # 打印訓練進度
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 早停機制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # 載入最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 評估階段
    model.eval()
    
    # 在測試集上進行預測
    with torch.no_grad():
        test_dataset = TensorDataset(test_x.to(device), test_y.to(device))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        test_preds = []
        test_trues = []
        
        for inputs, targets in test_loader:
            outputs = model(inputs)
            test_preds.append(outputs.cpu().numpy())
            test_trues.append(targets.cpu().numpy())
            
    # 組合所有批次的結果
    test_preds = np.concatenate(test_preds, axis=0)
    test_trues = np.concatenate(test_trues, axis=0)
    
    # 計算評估指標
    test_mae, test_mse, test_rmse, test_mape, test_mspe, test_rse, test_corr = evaluate_metrics(test_preds, test_trues)
    
    print(f"{model_name} 測試集評估結果:")
    print(f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.4f}, MSPE: {test_mspe:.4f}, RSE: {test_rse:.4f}, CORR: {test_corr:.4f}")
    
    return model, test_preds, test_trues, (test_mae, test_mse, test_rmse, test_mape, test_mspe, test_rse, test_corr)


if __name__ == "__main__":
    # 參數設置 - 與 weather.sh 保持一致
    seq_len = 96
    pred_lengths = [96, 192, 336, 720]  # 與 weather.sh 保持一致的預測長度
    random_seeds = [2024, 2025, 2026, 2027, 2028]  # 與 weather.sh 保持一致的隨機種子
    features = 'M'  # 使用多變量預測
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    batch_size = 256  # 與 weather.sh 一致
    learning_rate = 0.005  # 與 weather.sh 一致
    train_epochs = 30  # 與 weather.sh 一致
    patience = 5  # 與 weather.sh 一致
    
    # 對每個預測長度和隨機種子組合進行訓練和評估
    for pred_len in pred_lengths:
        print(f"\n===== 處理預測長度: {pred_len} =====")
        
        # 載入天氣數據
        print(f"載入天氣數據，序列長度={seq_len}，預測長度={pred_len}")
        train_x, train_y, val_x, val_y, test_x, test_y, scaler = load_weather_data(
            seq_len=seq_len, pred_len=pred_len, features=features
        )
        
        # 獲取特徵維度
        input_dim = train_x.shape[2]
        
        print(f"訓練數據形狀: {train_x.shape}, {train_y.shape}")
        print(f"驗證數據形狀: {val_x.shape}, {val_y.shape}")
        print(f"測試數據形狀: {test_x.shape}, {test_y.shape}")
        
        for seed in random_seeds:
            print(f"\n----- 使用隨機種子: {seed} -----")
            
            # 創建和訓練MLP模型
            mlp_model = MLP(seq_len, pred_len, input_dim, hidden_dims=[128, 256, 128])
            mlp_model, mlp_preds, mlp_trues, mlp_metrics = train_and_evaluate_model(
                mlp_model, "MLP", train_x, train_y, val_x, val_y, test_x, test_y, 
                batch_size=batch_size, epochs=train_epochs, learning_rate=learning_rate, 
                patience=patience, device=device, random_seed=seed
            )
            
            # 保存結果
            setting = f"weather_{seq_len}_{pred_len}_seed{seed}"
            save_nn_results(mlp_preds, mlp_trues, setting, "MLP", seq_len, pred_len)
            
            print(f"\n隨機種子 {seed} 的結果:")
            print(f"MLP\t\tMAE: {mlp_metrics[0]:.4f}\tMSE: {mlp_metrics[1]:.4f}\tRMSE: {mlp_metrics[2]:.4f}\tCORR: {mlp_metrics[6]:.4f}")

    print("\n所有實驗完成！")