import os
import pickle
import argparse
from sympy import Line2D
import torch
import random
import numpy as np
from tqdm import tqdm
from end_point import *
from utils import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pypots.optim.adam import Adam
#设置了固定随机种子确保可重复结果、禁用 CUDA 的非确定性优化与 TensorFloat32，从而保证模型可复现
# Reproducibility
torch.manual_seed(1729)
random.seed(1729)
np.random.seed(1729)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Argument parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=2) #输入特征维度（行人坐标 x, y）
parser.add_argument('--output_size', type=int, default=5) #输出特征维度，用于多混合高斯建模控制点
parser.add_argument('--n_epgcn', type=int, default=1,
                    help='Number of EPGCN layers for endpoint prediction')#端点预测模块的图卷积神经网络层数，1层足够处理多智能体间交互
parser.add_argument('--n_epcnn', type=int, default=6, 
                    help='Number of EPCNN layers for endpoint prediction') #端点预测模块的卷积神经网络层数，6层提供足够的体征提取能力 
parser.add_argument('--n_trgcn', type=int, default=1,
                    help='Number of TRGCN layers for trajectory refinement') #轨迹精炼模块的图卷积神经网络数，1层用于初步轨迹优化
parser.add_argument('--n_trcnn', type=int, default=3,
                    help='Number of TRCNN layers for trajectory refinement')#轨迹精炼模块的卷积神经网络数目，3层实现精细轨迹调整
parser.add_argument('--n_ways', type=int, default=3,
                    help='Number of control points for endpoint prediction') #端点预测的中间控制点数量
parser.add_argument('--n_smpl', type=int, default=20,
                    help='Number of samples for refine')#精炼阶段采样次数（用于多样化轨迹生成）
parser.add_argument('--kernel_size', type=int, default=3) #卷积核大小

# Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8) #观察序列长度，对应3.2秒的时间窗口
parser.add_argument('--pred_seq_len', type=int, default=12) #预测序列长度，对应4.8秒的时间窗口
parser.add_argument('--dataset', default='zara1',
                    help='Dataset name(eth,hotel,univ,zara1,zara2)') #数据集 校园开放区域，行人密度适中 / 酒店前院场景，空间相对受限 / 大学校园环境，包含复杂交互模式  商业街环境，高密度人群

# Training specifc parameters
parser.add_argument('--batch_size', type=int,
                    default=128, help='Mini batch size')  #训练批次大小，每次迭代处理的样本数量
parser.add_argument('--num_epochs', type=int,
                    default=512, help='Number of epochs') #训练总轮数，模型完整遍历数据集的次数
parser.add_argument('--clip_grad', type=float,
                    default=None, help='Gradient clipping')#梯度裁剪阈值，防止梯度爆炸，当梯度范数超过阈值时按比例缩放，保持梯度方向不变
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate') #学习率，控制参数更新步长
parser.add_argument('--lr_sh_rate', type=int, default=128,
                    help='Number of steps to drop the lr') #学习率下降步数，每128步降低学习率
parser.add_argument('--use_lrschd', action="store_true",
                    default=False, help='Use lr rate scheduler')#启用学习率调度器
parser.add_argument('--tag', default='tag', help='Personal tag for the model') #模型训练标识（用于日志命名等）
parser.add_argument('--nans', default=0.1, type=float, help='Number of nans prior to training') #缺失值比例（模拟传感器缺失情况），训练前引入10%的NaN值
parser.add_argument('--saits_lr', default=1e-4, type=float, help='Learning Rate of pre-trained SAITS model')#SAITS（插补模型）微调阶段的学习率，在端到端微调阶段对预训练组件使用独立学习率
args = parser.parse_args()

plt.figure(figsize=(20,20))
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layecliprs in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []

    # for n, p in named_parameters:
    #     print("parameters:", n,p)
    i = 0
    for n, p in named_parameters:
        i += 1
        if(p.requires_grad) and ("bias" not in n):
            # print(f'{type(p.grad)=}')
            # print(n)
            # if p.grad != None:
            if p.grad == None:
                #print(i, n, p)
                continue
            layers.append(n)
            ave_grads.append(p.grad.cpu().abs().mean())
            # max_grads.append(p.grad.abs().max())
    plt.plot(ave_grads, alpha=0.3, color="b")
    # plt.plot(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    # plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f"img/gradient-hotel")

#由于每个场景中的行人数不同，图结构顶点数随时间变化，因此批大小（batch_size）设为 1，以保证每次迭代中模型能正确处理变长输入
# Data preparation
# Batch size set to 1 because vertices vary by humans in each scene sequence.
# Use mini batch working like batch.
dataset_path = './datasets/' + args.dataset + '/'
checkpoint_dir = './checkpoint/' + args.tag + '/'

train_dataset = TrajectoryDataset(
    dataset_path + 'train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
train_loader = DataLoader(train_dataset, batch_size=1,
                          shuffle=True, num_workers=0, pin_memory=True)#pin_memory=True 以加快 GPU 数据传输，训练阶段启用随机打乱（shuffle=True）

val_dataset = TrajectoryDataset(
    dataset_path + 'val/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1)
val_loader = DataLoader(val_dataset, batch_size=1,
                        shuffle=False, num_workers=0, pin_memory=True)

# Model preparation
model = end_point(n_epgcn=args.n_epgcn, n_epcnn=args.n_epcnn, n_trgcn=args.n_trgcn, n_trcnn=args.n_trcnn,
                   seq_len=args.obs_seq_len, pred_seq_len=args.pred_seq_len, n_ways=args.n_ways, n_smpl=args.n_smpl)
model = model.to(device)
saits = create_saits_model(epochs=128) #插补模型（SAITS，Self-Attention-based Imputation Transformer）作为外部预训练模块加载，并与 MS-TIP 模型联合优化,该设计允许端到端训练中同时更新插补模块与预测模块的权重
all_parameters = list(model.parameters()) + list(saits.model.parameters())

optimizer = torch.optim.SGD(all_parameters, lr=args.lr)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=(1e-5)/2)
if args.use_lrschd:
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_sh_rate, gamma=0.8)#若启用学习率调度器（--use_lrschd），则每训练 128 步后学习率按衰减因子 γ=0.8 自动下降。
#每次训练运行均自动创建独立的检查点目录（./checkpoint/[tag]/），
#并保存模型参数与运行配置（通过 pickle 序列化）。
#这使得不同实验配置（如缺失值比例、学习率策略等）可以独立追踪与复现
# Train logging
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir + f'args-{args.nans}.pkl', 'wb') as f:
    pickle.dump(args, f)

metrics = {'train_loss': [], 'val_loss': []}#同时定义了训练与验证损失日志容器
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e10}


def transform_imputed(X):

    X_abs = X[:, :, :2]
    X_abs = X_abs.reshape(1, *X_abs.shape)

    X_rel = X[:, :, 2:]
    X_rel = X_rel.reshape(1, *X_rel.shape)

    X_abs = X_abs.permute(0, 1, 3, 2)
    X_rel = X_rel.permute(0, 1, 3, 2)

    S_obs = torch.stack((X_abs, X_rel), dim=1).permute(0, 1, 4, 2, 3)

    return S_obs


def saits_loader(original_tensor, nans = 0.1):
    nelems = original_tensor.numel()
    ne_nan = int(nans * nelems)
    nan_indices = random.sample(range(nelems), ne_nan)
    new_tensor = original_tensor.clone().reshape(-1)
    new_tensor[nan_indices] = float('nan')
    return new_tensor.reshape(*original_tensor.shape)


def train(epoch, nan=0):
    global metrics, model, saits
    model.train()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(train_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description(
        'Train Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))
    optimizer.zero_grad()
    for batch_idx, batch in enumerate(train_loader):
        # sum gradients till idx reach to batch_size
        if batch_idx % args.batch_size == 0:
            optimizer.zero_grad()

        S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]
        # print(f"{S_obs[:, 1]=}")
        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)

        S_actual = S_obs.clone()
        X_obs_saits = S_obs[:, 0].clone().to(device).permute(0, 2, 3, 1)
        X_obs_rel_saits = S_obs[:, 1].clone().to(device).permute(0, 2, 3, 1)


        _, npeds, _, step_size = X_obs_saits.shape
        X_obs_saits = X_obs_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        _, npeds, _, step_size = X_obs_rel_saits.shape
        X_obs_rel_saits = X_obs_rel_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        for i in range(npeds):
            X_i = X_obs_saits[i]
            X_obs_saits[i] = saits_loader(X_i, nan)

        for i in range(npeds):
            X_i = X_obs_rel_saits[i]
            X_obs_rel_saits[i] = saits_loader(X_i, nan)

        X_saits = torch.cat((X_obs_saits, X_obs_rel_saits), dim=2)
        X_obs_saits = saits_impute(X_saits)
        S_obs_imputed = transform_imputed(X_obs_saits)
        absolute_diff = torch.abs(S_obs_imputed - S_obs)
        mae_diff = torch.max(absolute_diff)
        S_obs = S_obs_imputed.clone()

        # Run Graph-TERN model
        # try:
        V_init, V_pred, V_refi, valid_mask = model(S_obs, S_trgt, vgg_list = vgg_list)
        # except:
        #     print(f"{S_actual=}")
        #     print(f"{S_obs_imputed=}")
        #     exit(1)
        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask)
        loss = r_loss + m_loss

        if torch.isnan(loss):
            pass
        else:
            loss.backward()
            # plot_grad_flow(model.named_parameters())
            loss_batch += loss.item()

        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)
            optimizer.step()

            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Train Epoch: {0} Loss: {1:.8f} Max_MAE: {2: .8f}'.format(
            epoch, loss.item() / args.batch_size, mae_diff))
        progressbar.update(1)

    progressbar.close()
    metrics['train_loss'].append(loss_batch / loader_len)


def valid(epoch):
    global metrics, constant_metrics, model, saits
    model.eval()
    loss_batch = 0.
    r_loss_batch, m_loss_batch = 0., 0.
    loader_len = len(val_loader)

    progressbar = tqdm(range(loader_len))
    progressbar.set_description(
        'Valid Epoch: {0} Loss: {1:.8f}'.format(epoch, 0))

    for batch_idx, batch in enumerate(val_loader):
        S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]

        # Run Graph-TERN model
        V_init, V_pred, V_refi, valid_mask = model(S_obs, vgg_list = vgg_list)

        # Loss calculation
        r_loss = gaussian_mixture_loss(V_init, S_trgt[:, 1], args.n_ways)
        m_loss = mse_loss(V_refi, S_trgt[:, 0], valid_mask, training=False)
        loss = r_loss + m_loss

        loss_batch += loss.item()
        r_loss_batch += r_loss.item()
        m_loss_batch += m_loss.item()

        if batch_idx % args.batch_size + 1 == args.batch_size or batch_idx + 1 == loader_len:
            r_loss_batch = 0.
            m_loss_batch = 0.

        progressbar.set_description('Valid Epoch: {0} Loss: {1:.8f}'.format(
            epoch, loss.item() / args.batch_size))
        progressbar.update(1)

    progressbar.close()
    metrics['val_loss'].append(loss_batch / loader_len)

    # Save model
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir +
                   args.dataset + str(args.nans) + f'_{epoch}_best.pth')
        torch.save(saits.model.state_dict(), checkpoint_dir+f'saits/{args.dataset}_{args.nans}_{epoch}_best.pth')

def get_dataset():
    tensors  = []
    # print(saits.optimizer)
    for batch_idx, batch in enumerate(train_loader):
        S_obs, S_trgt, vgg_list = [tensor.to(device) for tensor in batch[-3:]]

        # Data augmentation
        aug = True
        if aug:
            S_obs, S_trgt = data_sampler(S_obs, S_trgt, batch=1)

        # print(S_obs[:,0].shape)

        X_obs_saits = S_obs[:, 0].clone().to(device).permute(0, 2, 3, 1)
        X_obs_rel_saits = S_obs[:, 1].clone().to(device).permute(0, 2, 3, 1)

        # print(f'{X_obs_saits.shape=}')

        _, npeds, _, step_size = X_obs_saits.shape
        X_obs_saits = X_obs_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        _, npeds, _, step_size = X_obs_rel_saits.shape
        X_obs_rel_saits = X_obs_rel_saits.permute(
            0, 1, 3, 2).reshape(npeds, step_size, -1)

        for i in range(npeds):
            X_i = X_obs_saits[i]
            X_obs_saits[i] = saits_loader(X_i)

        for i in range(npeds):
            X_i = X_obs_rel_saits[i]
            X_obs_rel_saits[i] = saits_loader(X_i)

        X_saits = torch.cat((X_obs_saits, X_obs_rel_saits), dim=2)
        tensors.append(X_saits)
    combined_dataset = torch.cat(tensors, dim=0) #输出统一的时序数据集（含绝对与相对特征）
    return combined_dataset

def pre_train_saits():
    saits_model(get_dataset())

def main():
    import os
    import pickle
    global saits
    saits_pkl = f'pre-train/saits-{args.dataset}-tune64-Adam-Scaled.pth'
    #在主训练开始前，首先对 SAITS（Self-Attention-based Imputation Transformer） 模块进行独立预训练，用以学习时序缺失模式的填补能力。
    if os.path.exists(saits_pkl):#若本地存在预训练权重文件 pre-train/saits-[dataset]-tune64-Adam-Scaled.pth，则直接加载权重
        saits.model.load_state_dict(torch.load(saits_pkl))
    else:
        pre_train_saits()#不存在，则调用 pre_train_saits() 函数执行预训练
        torch.save(saits.model.state_dict(), saits_pkl)#并保存权重文件以便后续重复实验使用
    #该设计减少了主模型训练的冷启动难度，使轨迹插补模块在初始阶段即可输出稳定的时空估计结果
    print("saits_lr",args.saits_lr)
    print("lr", args.lr)
    saits.optimizer = Adam(lr = args.saits_lr, weight_decay= args.saits_lr/20)#SAITS 插补子模块使用独立的 Adam 优化器，学习率为 saits_lr=1×10⁻⁴，并施加权重衰减项 weight_decay=saits_lr/20
    saits.model.train() 
    #使预训练模块参数在端到端训练中保持稳定微调，防止梯度爆炸或知识遗忘

    # init_params = {}
    
    # for n,p in saits.model.named_parameters():
    #     init_params[n] = p.clone()
    #端到端训练与验证流程（End-to-End Training and Validation Loop）
    for epoch in range(args.num_epochs):
        train(epoch) #执行完整的端到端训练流程
        train(epoch, args.nans) #模拟不同缺失率条件下的插补与预测
        # curr_params={}
        # for n,p in saits.model.named_parameters():
        #     curr_params[n] = p

        # for key in init_params:
        #     print(init_params[key])
        #     print(curr_params[key])
        #     break
        # print(init_params)
        # print(curr_params)
        # are_equal = all(torch.equal(init_params[key], curr_params[key]) for key in init_params)
        # assert are_equal
        valid(epoch) #在验证集上计算 ADE/FDE 损失指标
        if args.use_lrschd:#若启用调度器（--use_lrschd），则每 128 步按比例衰减学习率（γ=0.8）
            scheduler.step()

        print(" ")
        print("Dataset: {0}, Epoch: {1}".format(args.tag, epoch))
        print("Train_loss: {0}, Val_los: {1}".format(
            metrics['train_loss'][-1], metrics['val_loss'][-1])) #每轮训练平均损失、验证集损失
        print("Min_val_epoch: {0}, Min_val_loss: {1}".format(
            constant_metrics['min_val_epoch'], constant_metrics['min_val_loss'])) #训练至当前的最优验证损失、对应的最优轮次
        print(" ")
        #所有实验均自动保存至独立的检查点目录 ./checkpoint/[tag]/，不同 tag 值可用于区分多组实验（如不同缺失比例或学习率设置），实验过程支持断点续训与结果复现
        with open(checkpoint_dir + f'metrics-{args.nans}.pkl', 'wb') as f:
            pickle.dump(metrics, f) #实验运行参数文件序列化保存

        with open(checkpoint_dir + f'constant_metrics-{args.nans}.pkl', 'wb') as f:
            pickle.dump(constant_metrics, f)#实验运行训练指标文件序列化保存


if __name__ == "__main__":
    main()
