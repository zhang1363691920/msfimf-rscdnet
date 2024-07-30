from options.train_options import TrainOptions
from data import create_dataset
from models.CDM import CDM
import os
import torch
from models.dice_loss import DiceLoss
from models.loss import CELoss
from util.metrics import AverageMeter, RunningMetrics
import time
import copy
from torchvision import utils as vutils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_val_opt(opt):

    val_opt = copy.deepcopy(opt)
    val_opt.preprocess = ''  #
    # hard-code some parameters for test
    val_opt.num_threads = 0   # test code only supports num_threads = 1
    val_opt.batch_size = opt.batch_size    # test code only supports batch_size = 1
    val_opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    val_opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    val_opt.angle = 0
    val_opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    val_opt.phase = 'val'
    val_opt.split = opt.val_split  # function in jsonDataset and ListDataset
    val_opt.isTrain = False
    val_opt.aspect_ratio = 1
    val_opt.dataroot = opt.val_dataroot
    val_opt.dataset_mode = opt.val_dataset_mode
    val_opt.dataset_type = opt.val_dataset_type
    val_opt.results_dir = './results/' + opt.name + '/' + val_opt.dataset_type
    val_opt.json_name = opt.val_json_name
    val_opt.eval = True

    return val_opt

def print_current_result(log_name, epoch, score, loss):
    """print current acc on console; also save the losses to the disk
    Parameters:
    """
    message = '(epoch: %d) ' % epoch
    message += 'loss: %.5f' % loss
    for k, v in score.items():
        message += '%s: %.5f ' % (k, v)
    print(message)  # print the message
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)  # save the message

def val(opt, model):
    # 创建数据集
    opt = make_val_opt(opt)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    model.eval()

    # loss_dice = DiceLoss().to(device)
    loss_ce = CELoss().to(device)

    # 创建验证结果保存文件
    val_log_name = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name, 'val_log.txt')
    with open(val_log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ val acc (%s) ================\n' % now)

    running_metrics = AverageMeter()
    total_val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataset):
            A = data['A'].to(device)
            B = data['B'].to(device)
            L = data['L'].to(device)
            # L_128 = data['L_128'].to(device)
            # L_64 = data['L_64'].to(device)
            # L_32 = data['L_32'].to(device)
            # L_16 = data['L_16'].to(device)
            # L_8 = data['L_8'].to(device)
            # change_map, s1, s2, s3, s4, s5 = model(A, B)
            change_map = model(A, B)
            pred = torch.argmax(change_map, dim=1)
            # pred_L = pred * 255
            # pred_L = pred_L.float()
            # vutils.save_image(pred_L, 'samples/LEVIR/val/output/'+str(i)+'/res.png')

            loss1 = loss_ce(change_map, L)
            # loss2 = loss_dice(change_map, L)
            # loss3 = loss_ce(s1, L_128)
            # loss4 = loss_ce(s2, L_64)
            # loss5 = loss_ce(s3, L_32)
            # loss6 = loss_ce(s4, L_16)
            # loss7 = loss_ce(s5, L_8)
            # loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 +loss7
            loss = loss1

            total_val_loss += loss.item()

            metrics = RunningMetrics(2)
            # print(L.size())
            # print(pred.size())
            metrics.update(L.detach().cpu().numpy(), pred.detach().cpu().numpy())
            scores = metrics.get_cm()

            running_metrics.update(scores)

        score = running_metrics.get_scores()
        print_current_result(val_log_name, epoch, score, total_val_loss / (dataset_size / opt.batch_size))

        return score['F1_1']

if __name__ == '__main__':
    # 准备训练集
    opt = TrainOptions().parse()
    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    # 设置网络训练的一些参数
    para_path = opt.name + '.pth'
    save_path = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name, para_path)
    best_f1 = 0.0
    total_loss = 0.0

    # 创建网络模型
    model = CDM().to(device)
    # print(model)

    # model.load_state_dict(torch.load('./checkpoints/LEVIR/CDM/23_90163(resnet50冻结).pth'))

    # 定义损失函数
    loss_dice = DiceLoss().to(device)
    loss_ce = CELoss().to(device)

    # 构造优化器
    # for name, param in model._modules.items():
    # if name == 'resnet':
    # for p in param.parameters():
    # p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)

    # 创建训练结果保存文件
    time_metric = AverageMeter()
    train_log_name = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name, 'train_log.txt')
    with open(train_log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ training time (%s) ================\n' % now)
    for epoch in range(opt.epoch):
        epoch_start_time = time.time()  # 每轮epoch的开始时间
        iter_data_time = time.time()  # timer for data loading per iteration
        total_iters = 0  # 当前epoch已经训练过的图片数
        epoch_total_loss = 0.0  # 每轮的loss

        # train
        model.train()
        for step, data in enumerate(dataset):
            # t1 = time.time()
            iter_start_time = time.time()  # 每一轮batch_size的开始时间
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            A = data['A'].to(device)
            B = data['B'].to(device)
            L = data['L'].to(device)
            # L_128 = data['L_128'].to(device)
            # L_64 = data['L_64'].to(device)
            # L_32 = data['L_32'].to(device)
            # L_16 = data['L_16'].to(device)
            # L_8 = data['L_8'].to(device)

            # forward
            # change_map, s1, s2, s3, s4, s5 = model(A, B)
            change_map = model(A, B)

            # print('change_map')
            # print(change_map.shape)
            # change_map = model(A, B, step, 'train', L)
            # pred = torch.argmax(change_map, dim=1)
            # pred_L = pred * 255
            # pred_L = pred_L.unsqueeze(1)
            # pred_L = pred_L.float()
            # vutils.save_image(pred_L, 'samples/LEVIR/train/output/' + str(step) + '/res.png')
            # zero_grad
            optimizer.zero_grad()
            # backward
            loss1 = loss_ce(change_map, L)
            # loss2 = loss_dice(change_map, L)
            # loss3 = loss_ce(s1, L_128)
            # loss4 = loss_ce(s2, L_64)
            # loss5 = loss_ce(s3, L_32)
            # loss6 = loss_ce(s4, L_16)
            # loss7 = loss_ce(s5, L_8)
            # loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7
            loss = loss1
            epoch_total_loss += loss.item()
            # step
            optimizer.step()
            # 打印损失函数并写入文件中
            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size

                message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f, loss: %f) ' % (
                epoch, total_iters, t_comp, t_data, loss.item())

                print(message)
                with open(train_log_name, "a") as log_file:
                    log_file.write('%s\n' % message)
        # 打印一个epoch花费的时间
        t_epoch = time.time() - epoch_start_time
        time_metric.update(t_epoch)
        print_current_result(train_log_name, epoch, {"current_t_epoch": t_epoch},epoch_total_loss / (dataset_size / opt.batch_size))
        total_loss += epoch_total_loss

        # validate
        val_f1 = val(opt, model)

        # 保存最优网络模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)

    time_ave = time_metric.average()
    print_current_result(train_log_name, opt.epoch, {"ave_t_epoch": time_ave},
                         total_loss / ((dataset_size / opt.batch_size) * opt.epoch))
