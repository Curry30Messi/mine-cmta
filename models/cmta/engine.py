import os
import numpy as np
from tqdm import tqdm

from sksurv.metrics import concordance_index_censored
from thop import profile, clever_format
import torch.optim
import torch.nn.parallel

from hypll.optim import RiemannianAdam
import matplotlib.pyplot as plt

def plot_loss_index(train_loss_all, train_index_all, val_loss_all, val_index_all, results_dir,fold):
    """
    绘制四张分别表示train和val的loss和index变化的图像，以及一张综合图。

    参数:
    - train_loss_all: 训练集的loss数据
    - train_index_all: 训练集的index数据
    - val_loss_all: 验证集的loss数据
    - val_index_all: 验证集的index数据
    - results_dir: 保存图像的目录

    """
    # 如果保存目录不存在，则创建
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 获取 epoch 数
    num_epochs = len(train_loss_all)
    epochs = list(range(1, num_epochs + 1))

    # 绘制前四张图
    plt.figure(figsize=(10, 8))

    # 第一张图：train_loss_all 随 epoch 变化
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_loss_all, label='Train Loss', color='blue')
    plt.ylim(1, 9)  # loss 范围
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epoch')
    plt.grid(True)

    # 第二张图：train_index_all 随 epoch 变化
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_index_all, label='Train Index', color='orange')
    plt.ylim(0.4, 0.8)  # index 范围
    plt.xlabel('Epoch')
    plt.ylabel('Train Index')
    plt.title('Train Index vs Epoch')
    plt.grid(True)

    # 第三张图：val_loss_all 随 epoch 变化
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_loss_all, label='Validation Loss', color='green')
    plt.ylim(1, 9)  # loss 范围
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs Epoch')
    plt.grid(True)

    # 第四张图：val_index_all 随 epoch 变化
    plt.subplot(2, 2, 4)
    plt.plot(epochs, val_index_all, label='Validation Index', color='red')
    plt.ylim(0.4, 0.8)  # index 范围
    plt.xlabel('Epoch')
    plt.ylabel('Validation Index')
    plt.title('Validation Index vs Epoch')
    plt.grid(True)

    # 保存前四张图到文件
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'loss_index_per_epoch__{fold}.png'))
    plt.close()

    # 绘制第五张图，将四条曲线绘制在同一张图中
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 左边的 y 轴 (loss)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(epochs, train_loss_all, label='Train Loss', color='blue', linestyle='--')
    ax1.plot(epochs, val_loss_all, label='Validation Loss', color='green', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_ylim(1, 9)  # loss 范围
    ax1.grid(True)

    # 右边的 y 轴 (index)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Index', color='red')
    ax2.plot(epochs, train_index_all, label='Train Index', color='orange')
    ax2.plot(epochs, val_index_all, label='Validation Index', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0.4, 0.8)  # index 范围

    # 添加图例
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))
    plt.title('Loss and Index vs Epoch')

    # 保存第五张图到文件
    plt.savefig(os.path.join(results_dir, f'combined_loss_index__{fold}.png'))
    plt.close()

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            self.validate(val_loader, model, criterion)
            return
        train_loss_all=[]
        train_index_all=[]
        val_loss_all=[]
        val_index_all=[]
        flops, params=0.0,0.0
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            train_loss,train_index,flops, params=self.train(train_loader, model, criterion, optimizer)
            if epoch == 0:
                print('flops:',flops,'params:',params)
            train_loss_all.append(train_loss)
            train_index_all.append(train_index)
            # evaluate on validation set
            c_index,val_loss,val_index = self.validate(val_loader, model, criterion, self.args.modality)
            val_loss_all.append(val_loss)
            val_index_all.append(val_index)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        plot_loss_index(train_loss_all, train_index_all, val_loss_all, val_index_all, self.results_dir,self.fold)
        return self.best_score, self.best_epoch,flops, params

    def random_mask_features(features, mask_prob):
        """
        :param features: (batch_size, num_feature, feature_dim)
        :param mask_prob: mask prob
        :return: masked
        """
        # 生成和features形状相同的二值掩码矩阵
        mask = torch.rand(features.shape) > mask_prob
        # 将特征值通过掩码矩阵乘以0，实现mask操作
        masked_features = features * mask.float()
        return masked_features

    def train(self, data_loader, model, criterion, optimizer):

        model.train()
        train_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        flops, params=0.0,0.0
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
                        c) in enumerate(dataloader):

            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            hazards, S, P, P_hat, G, G_hat,MLoss = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                   x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,
                                                   x_omic6=data_omic6)

            # survival loss + sim loss + sim loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            sim_loss_P = criterion[1](P.detach(), P_hat)
            sim_loss_G = criterion[1](G.detach(), G_hat)
            loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)


            if self.args.MoELoss:
                loss+=self.args.LossRate*MLoss
            # print("======================train==================")
            # print("loss:",loss)
            # print("sur_loss:",sur_loss)
            # print("self.args.alpha * (sim_loss_P + sim_loss_G)",self.args.alpha * (sim_loss_P + sim_loss_G))
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time
            train_loss += loss.item()

            # =======================================
            euclidean_params = [p for name, p in model.named_parameters() if 'hyperbolic' not in name]
            hyperbolic_params = [p for name, p in model.named_parameters() if 'hyperbolic' in name]
            #
            # # 定义优化器
            # optimizer_euclidean = torch.optim.SGD(filter(lambda p: p.requires_grad, euclidean_params), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
            # 假设 optimizer 已经定义
            # 首先清空 optimizer 的 param_groups
            optimizer.param_groups.clear()

            # 然后只将 euclidean_params 添加到 optimizer 中
            optimizer.add_param_group({'params': euclidean_params})

            # optimizer_euclidean = torch.optim.Adam(
            #     filter(lambda p: p.requires_grad, euclidean_params),
            #     lr=self.args.lr,
            #     weight_decay=self.args.weight_decay
            # )

            optimizer_hyperbolic = RiemannianAdam(hyperbolic_params, lr=0.001)
            # =======================================


            # for name, parms in model.named_parameters():
            #     if parms.grad is not None:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #               ' -->grad_value:', torch.mean(parms.grad))
            #     else:
            #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #               ' -->grad_value: None', )
            optimizer.zero_grad()
            optimizer_hyperbolic.zero_grad()


            loss.backward()

            optimizer.step()
            optimizer_hyperbolic.step()



            if batch_idx == len(data_loader) - 1:
                input_data = (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6)
                flops, params = profile(model, inputs=input_data)
                flops, params = clever_format([flops, params], "%.3f")
                # print(f"FLOPs: {flops}, Parameters: {params}")

            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('train loss: {:.4f}, c_index: {:.4f}'.format(train_loss, c_index))

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)
        return train_loss,c_index,flops, params

    def validate(self, data_loader, model, criterion,modality):

        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time,
                        c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI = data_WSI.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()
                if modality == 'Both':
                    pass
                if modality == 'G':
                    data_omic1 = torch.zeros_like(data_omic1).cuda()
                    data_omic2 = torch.zeros_like(data_omic2).cuda()
                    data_omic3 = torch.zeros_like(data_omic3).cuda()
                    data_omic4 = torch.zeros_like(data_omic4).cuda()
                    data_omic5 = torch.zeros_like(data_omic5).cuda()
                    data_omic6 = torch.zeros_like(data_omic6).cuda()
                if modality == 'P':
                    data_WSI = torch.zeros_like(data_WSI).cuda()



            with torch.no_grad():
                hazards, S, P, P_hat, G, G_hat ,MLoss= model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2,
                                                       x_omic3=data_omic3,
                                                       x_omic4=data_omic4, x_omic5=data_omic5,
                                                       x_omic6=data_omic6)  # return hazards, S, Y_hat, A_raw, results_dict

            # survival loss + sim loss + sim loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)
            sim_loss_P = criterion[1](P.detach(), P_hat)
            sim_loss_G = criterion[1](G.detach(), G_hat)
            loss = sur_loss + self.args.alpha * (sim_loss_P + sim_loss_G)
            if self.args.MoELoss:
                loss+=self.args.LossRate*MLoss
            # print("======================validate==================")
            # print("loss:",loss)
            # print("sur_loss:",sur_loss)
            # print("self.args.alpha * (sim_loss_P + sim_loss_G)",self.args.alpha * (sim_loss_P + sim_loss_G))
            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] = event_time
            val_loss += loss.item()

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('test loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))

        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
        return c_index, val_loss,c_index


    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'],
                                                                                          epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
