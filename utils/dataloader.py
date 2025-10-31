# Dataloader for pedestrian trajectory prediction benchmark.
# Sourcecode directly referred from Social-GAN at https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py

import os
import math
import random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1] ##对每条轨迹执行二次多项式拟合（degree=2），并根据残差误差判断其是否为非线性轨迹
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0 ## 非线性
    else:
        return 0.0


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002, min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len #8
        self.pred_len = pred_len #12
        self.skip = skip #1
        self.seq_len = self.obs_len + self.pred_len  #20
        self.delim = delim # '\t'

        # all_files = sorted(os.listdir(self.data_dir))
        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        all_files = [os.path.join(data_dir, path) for path in os.listdir(data_dir) if path[0] != "." and path.endswith(".txt")]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        fet_map = {}
        fet_list = []
        for path in all_files:
            print(path)
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist() #提取第一列的帧数,对每个文件按帧组织数据
            hkl_path = os.path.splitext(path)[0] + ".pkl"  
            with open(hkl_path, 'rb') as handle:
                new_fet = pickle.load(handle)
            fet_map[hkl_path] = torch.from_numpy(new_fet) #每个轨迹文件对应一个同名的 .pkl 文件，其中存储由预训练视觉模型（如 VGG-19）提取的场景语义特征。加载后映射至 fet_map,该特征随后输入至 MS-TIP 模型的 场景注意力模块（Scenic Attention Module），用于融合空间语义与轨迹信息。
            frame_data = []
            for frame in frames: #将同一文件中的所有帧按时间顺序组织起来
                frame_data.append(data[frame == data[:, 0], :]) #循环遍历所有唯一帧ID，按帧组织行人轨迹数据
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip)) #从连续帧数据中能够生成多少个有效的训练序列样本,skip为步长

            for idx in range(0, num_sequences * self.skip + 1, skip): #步长 skip 滑动窗口提取连续 seq_len 帧的子序列
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0) #(85,4)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1]) #8
                self.max_peds_in_frame = max(
                    self.max_peds_in_frame, len(peds_in_curr_seq)) #8

                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len)) #(8,2,20) #序列中的行人总数，坐标维度，序列长度(时间步长) 存储行人的绝对坐标序列
                curr_seq_rel = np.zeros(
                    (len(peds_in_curr_seq), 2, self.seq_len)) #存储行人的相对位移序列,用于学习速度与方向变化模式

                curr_loss_mask = np.zeros(
                    (len(peds_in_curr_seq), self.seq_len)) #(8,20) #损失掩码初始化，负责标记序列中每个行人在各个时间步的有效观测状态

                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq): #遍历当前时间窗口中出现的每个行人，提取其轨迹。
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1]
                                                 == ped_id, :] #行人id
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len: #过滤掉轨迹长度不足 seq_len 的行人（保证完整轨迹）
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    # curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:,
                                                           1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered

                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append( 
                        poly_fit(curr_ped_seq, pred_len, threshold)) #该标注结果（non_linear_ped）在后续损失计算中用于加权处理，使模型在学习曲线运动（如转弯、避让）时更加敏感
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    fet_list.append(hkl_path)


        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        self.fet_map = fet_map
        self.fet_list = fet_list
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.non_linear_ped = torch.from_numpy(
            non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(
            cum_start_idx, cum_start_idx[1:])]

        self.missing_obs_traj = self.saits_loader(self.obs_traj)
        self.missing_pred_traj = self.saits_loader(self.pred_traj)

        # Convert Trajectories to Graphs
        self.S_obs = []
        self.S_trgt = []

        pbar = tqdm(total=len(self.seq_start_end))
        pbar.set_description('Processing {0} dataset {1}'.format(
            self.data_dir.split('/')[-3], self.data_dir.split('/')[-2]))

        for i in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[i]
            s_obs = torch.stack(
                [self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :]], dim=0).permute(0, 3, 1, 2)#为支持后续的超图建模，加载器将轨迹序列转换为图结构张量
            self.S_obs.append(s_obs.clone())
            s_trgt = torch.stack(
                [self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :]], dim=0).permute(0, 3, 1, 2)
            self.S_trgt.append(s_trgt.clone())#输出结果 S_obs 与 S_trgt 分别对应观测阶段与预测阶段的图结构输入，每个图节点表示一个行人，边表示同一时间步的空间关系，构成模型中多尺度超图的基础
            pbar.update(1)
        pbar.close()


    def saits_loader(self, original_tensor):#为训练插补模块（SAITS）,该机制确保模型能在不完整轨迹输入下学习稳定的插补与预测
        nelems = original_tensor.numel()
        ne_nan = int(0.20 * nelems)#数据加载器随机将 20% 的输入轨迹值替换为 NaN（即模拟传感器遮挡或追踪丢失）
        nan_indices = random.sample(range(nelems), ne_nan)
        new_tensor = original_tensor.clone()
        new_tensor.view(-1)[nan_indices] = float('nan')

        return new_tensor

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.missing_obs_traj[start:end, :], self.missing_pred_traj[start:end, :], #带缺失值的观测序列（输入给 SAITS）, 带缺失值的预测序列（训练一致性）
            self.obs_traj[start:end, :], self.pred_traj[start:end, :], #原始观测轨迹，真实未来轨迹
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],#相对位移序列，预测位移序列
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],#非线性轨迹标志（0=线性，1=非线性）， 损失掩码，指示有效观测步
            self.S_obs[index], self.S_trgt[index],#观测阶段的图结构张量，预测阶段的图结构张量
            self.fet_map[self.fet_list[index]] #场景语义特征（VGG 提取）
        ]
        return out
