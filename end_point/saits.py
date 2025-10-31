# Install PyPOTS first: pip install pypots==0.1.1
import numpy as np
from sklearn.preprocessing import StandardScaler
from pypots.data import load_specific_dataset, mcar, masked_fill
from pypots.imputation import SAITS
from pypots.utils.metrics import cal_mae
from pypots.optim.adam import Adam
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_saits_model(n_steps=8, n_features=4, n_layers=2, d_model=64, d_inner=32, n_heads=4, d_k=16, d_v=16, dropout=0.1, epochs=100):
    global saits #时间步长度（序列长度），每个时间步的输入特征维度（如坐标及派生量），Transformer 编码层数，Transformer 隐层维度，前馈层维度，多头注意力头数，每个注意力头的键/值维度，随机失活比例，预训练轮数
    saits = SAITS(n_steps=n_steps, n_features=n_features, n_layers=n_layers, d_model=d_model, d_inner=d_inner, n_heads=n_heads, d_k=d_k, d_v=d_v, dropout=dropout, epochs=epochs, optimizer=Adam(lr=1e-4, weight_decay=(1e-5)/2))
    return saits

def saits_model(X):#(B,T:8,D) (批量大小，时间步8帧观测，特征维度2 x,y)
    global saits
    # Reshape the tensor to 2D (40 x 2)
    tensor_2d = X.reshape(-1, saits.n_features).cpu().numpy()
    # Create a StandardScaler object and fit it to the data
    scaler = StandardScaler()
    scaled_tensor_2d = scaler.fit_transform(tensor_2d)
    # Reshape the scaled NumPy array back to the original shape (5 x 8 x 2)
    X = torch.tensor(scaled_tensor_2d, dtype=torch.float32).view(*X.shape)

    # print(f'{saits.model.parameters()}')
    #缺失机制模拟 采用随机缺失机制 MCAR随机屏蔽 10% 的输入值，并生成缺失掩码 missing_mask 与真实掩码 indicating_mask 模型在仅观察到部分轨迹点的情况下学习恢复完整序列
    X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
    X = masked_fill(X, 1 - missing_mask, np.nan)
    #模型训练与插补
    dataset = {"X": X}
    saits.fit(dataset)  #SAITS 模型在训练阶段最小化插补误差，通过掩码控制反向传播范围
    # train the model. Here I use the whole dataset as the training set, because ground truth is not visible to the model.
    # imputation = saits.impute(dataset)
    # imputation = torch.from_numpy(imputation)
    # imputation = imputation.to(device)
    # X_intact = X_intact.to(device)
    # indicating_mask = indicating_mask.to(device)
    # mae = cal_mae(imputation, X_intact, indicating_mask)
    # return imputation, mae

def saits_impute(X):
    # X_intact, X, missing_mask, indicating_mask = mcar(X, 0.1) # hold out 10% observed values as ground truth
    tensor_2d = X.reshape(-1, saits.n_features).cpu().numpy()
    # Create a StandardScaler object and fit it to the data
    scaler = StandardScaler()
    scaled_tensor_2d = scaler.fit_transform(tensor_2d)

    # Reshape the scaled NumPy array back to the original shape (5 x 8 x 2)
    X_scaled = torch.tensor(scaled_tensor_2d, dtype=torch.float32).view(*X.shape)

    missing_mask = (~torch.isnan(X_scaled)).to(torch.float32)
    # X = masked_fill(X, 1 - missing_mask, np.nan)

    missing_mask = missing_mask.reshape(*X.shape)
    data = [X_scaled, missing_mask]
    X_scaled, missing_mask = map(lambda x: x.to(device), data)

    #在推理阶段，使用 saits.model.forward(data, training=False) 生成插补结果

    data = {
        "X": torch.nan_to_num(X_scaled, nan=0.0),
        "missing_mask": missing_mask,
    }
    results = saits.model.forward(data, training=False)
    imputation = results["imputed_data"]
    std = torch.from_numpy(scaler.scale_).to(device)
    mean = torch.from_numpy(scaler.mean_).to(device)
    # print(f"{std=}")
    # print(f"{mean=}")
    #插补输出在标准化空间中，因此需恢复原始尺度
    imputation *= std
    imputation += mean
    # imputation = saits.impute(dataset)
    # imputation = torch.from_numpy(imputation)
    imputation = imputation.to(device)
    abs_tensor = torch.abs(imputation)
    #随后对数值极小（<1e-5）的结果置零，以减少数值噪声
    imputation = torch.where(abs_tensor < 1e-5, torch.tensor(0.0).to(device), imputation) 
    #最终得到无缺失的轨迹张量，供主模型的多尺度超图模块与预测模块使用
    # X = X.to(device)
    # indicating_mask = indicating_mask.to(device)
    # mae = cal_mae(imputation, X_intact, indicating_mask)
    return imputation
'''
缺失值恢复：在存在遮挡或传感器故障的场景下重建轨迹；

时序一致性建模：通过自注意力机制建模全局时间依赖；

可微端到端优化：与 MS-TIP 主干网络联合训练，保持梯度可传播；

跨数据集泛化：通过随机缺失模拟提升鲁棒性。
'''