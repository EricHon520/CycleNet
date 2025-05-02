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
from utils.tools import EarlyStopping
from ml_algorithms import set_seed  
warnings.filterwarnings('ignore')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def load_weather_data(root_path='./dataset/', data_path='weather.csv', 
                   seq_len=96, pred_len=96, features='M'):
    df_raw = pd.read_csv(os.path.join(root_path, data_path))

    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    
    border1s = [0, num_train - seq_len, len(df_raw) - num_test - seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]

    if features == 'M' or features == 'MS':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]
    else:
        target = 'OT'
        df_data = df_raw[[target]]

    scaler = StandardScaler()
    train_data = df_data[border1s[0]:border2s[0]]
    scaler.fit(train_data.values)
    data = scaler.transform(df_data.values)

    df_stamp = df_raw[['date']][border1s[0]:border2s[2]]
    df_stamp['date'] = pd.to_datetime(df_stamp.date)
    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
    data_stamp = df_stamp.drop(columns=['date']).values

    train_x, train_y = prepare_data_for_nn(data[border1s[0]:border2s[0]], seq_len, pred_len)
    val_x, val_y = prepare_data_for_nn(data[border1s[1]:border2s[1]], seq_len, pred_len)
    test_x, test_y = prepare_data_for_nn(data[border1s[2]:border2s[2]], seq_len, pred_len)

    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    val_x = torch.FloatTensor(val_x)
    val_y = torch.FloatTensor(val_y)
    test_x = torch.FloatTensor(test_x)
    test_y = torch.FloatTensor(test_y)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, scaler


def prepare_data_for_nn(data, seq_len, pred_len):
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
    pred = np.array(pred)
    true = np.array(true)
    
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    rmse = np.sqrt(mse)
    
    mask = true != 0
    mape = np.mean(np.abs((true[mask] - pred[mask]) / true[mask]))
    mspe = np.mean(((true[mask] - pred[mask]) / true[mask]) ** 2)

    mean_true = np.mean(true)
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - mean_true) ** 2)
    rse = numerator / denominator if denominator != 0 else np.inf

    pred_flat = pred.reshape(-1)
    true_flat = true.reshape(-1)
    corr = np.corrcoef(pred_flat, true_flat)[0, 1] if len(pred_flat) > 1 else 0
    
    return mae, mse, rmse, mape, mspe, rse, corr


def save_nn_results(pred, true, setting, model_name, seq_len, pred_len):
    folder_path = f'./MLP_LSTM_GRU_results/{setting}_{model_name}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    mae, mse, rmse, mape, mspe, rse, corr = evaluate_metrics(pred, true)

    print(f'{model_name} - mse:{mse:.4f}, mae:{mae:.4f}')
    with open("result.txt", 'a') as f:
        f.write(f"{setting}_{model_name}  \n")
        f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}, rse:{rse:.4f}, corr:{corr:.4f}')
        f.write('\n\n')

    np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
    np.save(folder_path + 'pred.npy', pred)
    np.save(folder_path + 'true.npy', true)
    
    visualize_predictions(pred, true, folder_path, seq_len, pred_len)
    
    return mae, mse, rmse, mape, mspe, rse, corr


def visualize_predictions(pred, true, folder_path, seq_len, pred_len, samples=5):
    n_samples = min(samples, len(pred))
    
    if len(pred.shape) > 2:
        n_vars = pred.shape[2]

        target_var = 0
        
        for i in range(n_samples):
            plt.figure(figsize=(12, 6))
            plt.plot(true[i, :, target_var], label='true_value', color='blue', linestyle='-')
            plt.plot(pred[i, :, target_var], label='predicted_value', color='red', linestyle='--')
            plt.xlabel('time_step')
            plt.ylabel('value')
            plt.legend()
            plt.title(f'Sample {i+1}: prediction vs true value (variable {target_var})')
            plt.grid(True)
            plt.savefig(os.path.join(folder_path, f'sample_{i+1}_var_{target_var}.png'), 
                      dpi=300, bbox_inches='tight')
            plt.close()
    else:
        for i in range(n_samples):
            plt.figure(figsize=(12, 6))
            plt.plot(true[i], label='true_value', color='blue', linestyle='-')
            plt.plot(pred[i], label='predicted_value', color='red', linestyle='--')
            plt.xlabel('time_step')
            plt.ylabel('value')
            plt.legend()
            plt.title(f'Sample {i+1}: prediction vs true value (variable {target_var})')
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
    
    plt.plot(true_mean, label='true_mean', color='blue', linestyle='-')
    plt.plot(pred_mean, label='predicted_mean', color='red', linestyle='--')
    plt.xlabel('time_step')
    plt.ylabel('value')
    plt.legend()
    plt.title('Average predictions of all samples vs true values')
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, 'average_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_predictions(true_vals, pred_vals, step_idx=0, var_idx=0, n_points=100, title="Predictions vs True Values"):

    if len(true_vals.shape) > 2:
        n_variables = true_vals.shape[2]

        true_to_plot = true_vals[:n_points, step_idx, var_idx]
        pred_to_plot = pred_vals[:n_points, step_idx, var_idx]
    else:  
        n_variables = true_vals.shape[1] // pred_len  
        idx = step_idx * n_variables + var_idx
        true_to_plot = true_vals[:n_points, idx]
        pred_to_plot = pred_vals[:n_points, idx]
    
    plt.figure(figsize=(10, 5))
    plt.plot(true_to_plot, label='True', marker='o', markersize=3)
    plt.plot(pred_to_plot, label='Predicted', marker='x', markersize=3)
    plt.legend()
    plt.title(title)
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.close()

class MLP(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim, hidden_dims=[64, 128, 64], seed=None):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.flatten_dim = seq_len * input_dim
        
        if seed is not None:
            torch.manual_seed(seed)

        layers = []
        input_size = self.flatten_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_dim
            
        self.output_layer = nn.Linear(input_size, pred_len * input_dim)
        
        self.hidden_layers = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.reshape(batch_size, self.pred_len, self.input_dim)

class LSTM(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim, hidden_dim=128, num_layers=2, dropout=0.1, seed=None):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # LSTM層
        output, (h_n, c_n) = self.lstm(x)
        
        # 使用最後一個時間步的隱藏狀態來預測未來
        h_out = h_n[-1].view(batch_size, 1, self.hidden_dim).repeat(1, self.pred_len, 1)
        
        # 全連接層
        out = self.fc(h_out)
        
        return out

class GRU(nn.Module):
    def __init__(self, seq_len, pred_len, input_dim, hidden_dim=128, num_layers=2, dropout=0.1, seed=None):
        super(GRU, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if seed is not None:
            torch.manual_seed(seed)
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # GRU層
        output, h_n = self.gru(x)
        
        # 使用最後一個時間步的隱藏狀態來預測未來
        h_out = h_n[-1].view(batch_size, 1, self.hidden_dim).repeat(1, self.pred_len, 1)
        
        # 全連接層
        out = self.fc(h_out)
        
        return out

def train_and_evaluate_model(model, model_name, train_x, train_y, val_x, val_y, test_x, test_y, 
                            batch_size=256, epochs=30, learning_rate=0.005, patience=5, device='cpu', random_seed=2024):

    print(f"Starting training {model_name} model...")
    
    set_seed(random_seed)
    
    model = model.to(device)
    
    train_dataset = TensorDataset(train_x.to(device), train_y.to(device))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_x.to(device), val_y.to(device))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    checkpoint_path = f'./checkpoints/{model_name}_seq{train_x.shape[1]}_pred{train_y.shape[1]}_seed{random_seed}'
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # 使用 EarlyStopping
        early_stopping(val_loss, model, checkpoint_path)
        if early_stopping.early_stop:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    # 載入最佳模型
    model.load_state_dict(torch.load(os.path.join(checkpoint_path, 'checkpoint.pth')))
    
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
    
    print(f"{model_name} Test set evaluation results:")
    print(f"MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")
    print(f"MAPE: {test_mape:.4f}, MSPE: {test_mspe:.4f}, RSE: {test_rse:.4f}, CORR: {test_corr:.4f}")
    
    return model, test_preds, test_trues, (test_mae, test_mse, test_rmse, test_mape, test_mspe, test_rse, test_corr)


if __name__ == "__main__":
    seq_len = 96
    pred_lengths = [96]  
    random_seeds = [2024]  
    features = 'M'  
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device used: {device}")
    batch_size = 256  
    learning_rate = 0.005 
    train_epochs = 30  
    patience = 5  
    
    for pred_len in pred_lengths:
        print(f"\n===== Handle prediction length: {pred_len} =====")
        
        print(f"Load weather data, sequence length = {seq_len}, prediction length = {pred_len}")
        train_x, train_y, val_x, val_y, test_x, test_y, scaler = load_weather_data(
            seq_len=seq_len, pred_len=pred_len, features=features
        )
        
        input_dim = train_x.shape[2]

        mlp_results = []
        lstm_results = []
        gru_results = []
        
        print(f"Training data shape: {train_x.shape}, {train_y.shape}")
        print(f"Validation data shape: {val_x.shape}, {val_y.shape}")
        print(f"Test data shape: {test_x.shape}, {test_y.shape}")
        
        for seed in random_seeds:
            print(f"\n----- Use random seed: {seed} -----")
            
            mlp_model = MLP(seq_len, pred_len, input_dim, hidden_dims=[128, 256, 128], seed=seed)
            mlp_model, mlp_preds, mlp_trues, mlp_metrics = train_and_evaluate_model(
                mlp_model, "MLP", train_x, train_y, val_x, val_y, test_x, test_y, 
                batch_size=batch_size, epochs=train_epochs, learning_rate=learning_rate, 
                patience=patience, device=device, random_seed=seed
            )
            
            lstm_model = LSTM(seq_len, pred_len, input_dim, hidden_dim=128, num_layers=2, seed=seed)
            lstm_model, lstm_preds, lstm_trues, lstm_metrics = train_and_evaluate_model(
                lstm_model, "LSTM", train_x, train_y, val_x, val_y, test_x, test_y, 
                batch_size=batch_size, epochs=train_epochs, learning_rate=learning_rate, 
                patience=patience, device=device, random_seed=seed
            )
            
            gru_model = GRU(seq_len, pred_len, input_dim, hidden_dim=128, num_layers=2, seed=seed)
            gru_model, gru_preds, gru_trues, gru_metrics = train_and_evaluate_model(
                gru_model, "GRU", train_x, train_y, val_x, val_y, test_x, test_y, 
                batch_size=batch_size, epochs=train_epochs, learning_rate=learning_rate, 
                patience=patience, device=device, random_seed=seed
            )
            
            setting = f"weather_{seq_len}_{pred_len}_seed{seed}"
            save_nn_results(mlp_preds, mlp_trues, setting, "MLP", seq_len, pred_len)
            save_nn_results(lstm_preds, lstm_trues, setting, "LSTM", seq_len, pred_len)
            save_nn_results(gru_preds, gru_trues, setting, "GRU", seq_len, pred_len)

            mlp_results.append({
                'seed': seed,
                'metrics': mlp_metrics,
                'preds': mlp_preds,
                'trues': mlp_trues
            })
            
            lstm_results.append({
                'seed': seed,
                'metrics': lstm_metrics,
                'preds': lstm_preds,
                'trues': lstm_trues
            })
            
            gru_results.append({
                'seed': seed,
                'metrics': gru_metrics,
                'preds': gru_preds,
                'trues': gru_trues
            })
            
            print(f"\nResults for random seed {seed}:")
            print(f"MLP\t\tMAE: {mlp_metrics[0]:.4f}\tMSE: {mlp_metrics[1]:.4f}\tRMSE: {mlp_metrics[2]:.4f}\tCORR: {mlp_metrics[6]:.4f}")
            print(f"LSTM\t\tMAE: {lstm_metrics[0]:.4f}\tMSE: {lstm_metrics[1]:.4f}\tRMSE: {lstm_metrics[2]:.4f}\tCORR: {lstm_metrics[6]:.4f}")
            print(f"GRU\t\tMAE: {gru_metrics[0]:.4f}\tMSE: {gru_metrics[1]:.4f}\tRMSE: {gru_metrics[2]:.4f}\tCORR: {gru_metrics[6]:.4f}")
        
        print(f"\n{'-'*50}")
        print(f"Average results across {len(random_seeds)} seeds:")
        
        avg_mlp_mae = np.mean([res['metrics'][0] for res in mlp_results])
        avg_mlp_mse = np.mean([res['metrics'][1] for res in mlp_results])
        avg_mlp_rmse = np.mean([res['metrics'][2] for res in mlp_results])
        avg_mlp_corr = np.mean([res['metrics'][6] for res in mlp_results])
        
        avg_lstm_mae = np.mean([res['metrics'][0] for res in lstm_results])
        avg_lstm_mse = np.mean([res['metrics'][1] for res in lstm_results])
        avg_lstm_rmse = np.mean([res['metrics'][2] for res in lstm_results])
        avg_lstm_corr = np.mean([res['metrics'][6] for res in lstm_results])
        
        avg_gru_mae = np.mean([res['metrics'][0] for res in gru_results])
        avg_gru_mse = np.mean([res['metrics'][1] for res in gru_results])
        avg_gru_rmse = np.mean([res['metrics'][2] for res in gru_results])
        avg_gru_corr = np.mean([res['metrics'][6] for res in gru_results])
        
        print(f"MLP\t\tMAE: {avg_mlp_mae:.4f}\tMSE: {avg_mlp_mse:.4f}\tRMSE: {avg_mlp_rmse:.4f}\tCORR: {avg_mlp_corr:.4f}")
        print(f"LSTM\t\tMAE: {avg_lstm_mae:.4f}\tMSE: {avg_lstm_mse:.4f}\tRMSE: {avg_lstm_rmse:.4f}\tCORR: {avg_lstm_corr:.4f}")
        print(f"GRU\t\tMAE: {avg_gru_mae:.4f}\tMSE: {avg_gru_mse:.4f}\tRMSE: {avg_gru_rmse:.4f}\tCORR: {avg_gru_corr:.4f}")
        
        mlp_best_idx = np.argmin([res['metrics'][0] for res in mlp_results])
        mlp_best_seed = mlp_results[mlp_best_idx]['seed']
        mlp_best_preds = mlp_results[mlp_best_idx]['preds']
        mlp_best_trues = mlp_results[mlp_best_idx]['trues']
        
        lstm_best_idx = np.argmin([res['metrics'][0] for res in lstm_results])
        lstm_best_seed = lstm_results[lstm_best_idx]['seed']
        lstm_best_preds = lstm_results[lstm_best_idx]['preds']
        lstm_best_trues = lstm_results[lstm_best_idx]['trues']
        
        gru_best_idx = np.argmin([res['metrics'][0] for res in gru_results])
        gru_best_seed = gru_results[gru_best_idx]['seed']
        gru_best_preds = gru_results[gru_best_idx]['preds']
        gru_best_trues = gru_results[gru_best_idx]['trues']
        
        plot_predictions(mlp_best_trues, mlp_best_preds, 
                        title=f"MLP Best Results (pred_len={pred_len}, seed={mlp_best_seed})")
                        
        plot_predictions(lstm_best_trues, lstm_best_preds, 
                        title=f"LSTM Best Results (pred_len={pred_len}, seed={lstm_best_seed})")
                        
        plot_predictions(gru_best_trues, gru_best_preds, 
                        title=f"GRU Best Results (pred_len={pred_len}, seed={gru_best_seed})")

    print("\nAll experiments completed!")