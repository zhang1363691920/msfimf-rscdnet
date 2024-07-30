import time
from options.train_options import TrainOptions
from data import create_dataset
import os
from util.metrics import AverageMeter, RunningMetrics
import copy
import torch
from models.CDM_Visual import CDM
from models.loss import CELoss
from models.dice_loss import DiceLoss
from torchvision import utils as vutils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    val_opt.dataroot = opt.test_dataroot
    val_opt.dataset_mode = opt.val_dataset_mode
    val_opt.dataset_type = opt.val_dataset_type
    val_opt.results_dir = './results/' + opt.name + '/' + val_opt.dataset_type
    val_opt.json_name = opt.val_json_name
    val_opt.eval = True

    return val_opt

def print_current_result(score, loss):
    """print current acc on console; also save the losses to the disk
    Parameters:
    """
    # message = '(epoch: %d) ' % epoch
    message = 'loss: %05f' % loss
    for k, v in score.items():
        message += '%s: %.5f ' % (k, v)
    print(message)  # print the message
    # with open(log_name, "a") as log_file:
    #     log_file.write('%s\n' % message)  # save the message

def _visualize_pred(self, change_map):
    pred = torch.argmax(change_map, dim=1, keepdim=True)
    pred_vis = pred * 255
    return pred_vis

def test(opt, model):
    # 创建数据集
    opt = make_val_opt(opt)
    dataset = create_dataset(opt)
    dataset_size = len(dataset)

    model.eval()
    loss_dice = DiceLoss().to(device)
    loss_ce = CELoss().to(device)

    # 创建验证结果保存文件
    # val_log_name = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name, 'val_log.txt')
    # with open(val_log_name, "a") as log_file:
    #     now = time.strftime("%c")
    #     log_file.write('================ val acc (%s) ================\n' % now)

    running_metrics = AverageMeter()
    total_test_loss = 0.0
    with torch.no_grad():
        star_index = 1
        for i, data in enumerate(dataset):
            A = data['A'].to(device)
            B = data['B'].to(device)
            L = data['L'].to(device)
            change_map = model(A, B, i, L)
            # change_map = model(A, B)
            pred = torch.argmax(change_map, dim=1, keepdim=True)
            # pred_L = pred * 255
            # pred_L = pred_L.float()
            #
            # output_dir = 'samples/LEVIR/test/output'
            # os.makedirs(output_dir, exist_ok=True)
            # for j in range(pred_L.shape[0]):
            #     file_name = os.path.join(output_dir, 'test_{}.png'.format(star_index))
            #     vutils.save_image(pred_L[j], file_name)
            #     star_index += 1
            # vutils.save_image(pred_L, 'samples/LEVIR/test/output/'+str(i)+'/res.png')
            loss1 = loss_ce(change_map, L)
            loss2 = loss_dice(change_map, L)
            loss = loss1 + loss2
            total_test_loss += loss.item()

            metrics = RunningMetrics(2)
            metrics.update(L.detach().cpu().numpy(), pred.detach().cpu().numpy())
            scores = metrics.get_cm()

            running_metrics.update(scores)
            #_visualize_pred(change_map)

    score = running_metrics.get_scores()
    print_current_result(score, total_test_loss/(dataset_size/opt.batch_size))


if __name__ == '__main__':
    # 准备数据集
    opt = TrainOptions().parse()

    # 设置训练网络的一些参数
    para_path = opt.name + '.pth'
    save_path = os.path.join(opt.checkpoints_dir, opt.dataset_type, opt.name, para_path)

    # 创建网络模型
    model = CDM().to(device)
    model.load_state_dict(torch.load(save_path))

    test_f1 = test(opt, model)
