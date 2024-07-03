import os
import re
from os.path import join

import matplotlib.pyplot as plt


def __getdata(save_path, file_name):
    loss_ls = []
    metrics_dict = {}
    with open(join(save_path, file_name), 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            # l1,l2 format:
            #   val, epoch: 0, loss=0.23398
            #   This epoch Acc: 0.500, Class Acc: 0.565, Sensitivity: 0.130, Specificity: 1.000, F1-Score: 0.231
            l1 = lines[i]
            l2 = lines[i + 1]

            loss = float(l1.split('=')[-1].strip())

            # 使用正则表达式提取信息
            pattern = r"(\w+ \w+|\w+[-\w]*): (\d\.\d{3})"
            matches = re.findall(pattern, l2)

            # 将提取的信息转换为字典
            if len(metrics_dict) == 0:
                metrics_dict = {match[0]: [float(match[1])] for match in matches}
            else:
                for match in matches:
                    metrics_dict[match[0]].append(float(match[1]))

            loss_ls.append(loss)

    return loss_ls, metrics_dict


def __expoint_plotting(x, y, maximum, expoint_position_ls):
    ex_point = max(y) if maximum else min(y)
    plt.plot(x[y.index(ex_point)], ex_point, 'ro')
    pos = [x[y.index(ex_point)] + 0.1, ex_point]
    for pre_pos in expoint_position_ls:
        if abs(pre_pos[0] - pos[0]) < 2. and abs(pre_pos[1] - pos[1]) < 0.05:
            pos[0] -= 1
    plt.annotate(f"{ex_point}",
                 xy=(x[y.index(ex_point)], ex_point),
                 xytext=(pos),
                 fontsize=16)
    expoint_position_ls.append(pos)


def plot_metrics(configs, file_name, metrics_ls):
    _, metrics_dict = __getdata(configs['save_path'], file_name)
    if not len(metrics_ls) == 0:
        metrics_dict = {key: metrics_dict[key] for key in metrics_ls if key in metrics_dict}

    plt.figure(figsize=(12,8))
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    title = 'model: %s, %s, epoch: %d, lr: %.5f, batch size: %d' % (configs['model_name'], file_name[:-4], configs['epoch'], configs['lr'], configs['batch_size'])
    plt.title(title)

    x = range(1, configs['epoch']+1)
    expoint_position_ls = []
    for metric_name, y in metrics_dict.items():
        plt.plot(x, y, label=metric_name)
        __expoint_plotting(x, y, maximum=True, expoint_position_ls= expoint_position_ls)
    plt.legend()
    #plt.show()
    path = os.path.join(configs['save_path'], file_name[:-4] + '_metrics.png')
    plt.savefig(path)
    plt.close()


def plot_loss(configs):
    loss_ls_train, _ = __getdata(configs['save_path'], 'train.txt')
    if configs['val']:
        loss_ls_val, _ = __getdata(configs['save_path'], 'val.txt')

    plt.figure(figsize=(12,8))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = 'model: %s, epoch: %d, lr: %.5f, batch size: %d' % (configs['model_name'], configs['epoch'], configs['lr'], configs['batch_size'])
    plt.title(title)

    x = range(1, configs['epoch'] + 1)
    expoint_position_ls = []
    plt.plot(x, loss_ls_train, label='train', color='b')
    __expoint_plotting(x, loss_ls_train, maximum=False, expoint_position_ls=expoint_position_ls)
    if configs['val']:
        plt.plot(x, loss_ls_val, label='val', color='r')
        __expoint_plotting(x, loss_ls_val, maximum=False, expoint_position_ls=expoint_position_ls)
    plt.legend()
    #plt.show()
    path = os.path.join(configs['save_path'], 'loss.png')
    plt.savefig(path)
    plt.close()


# if __name__ == '__main__':
#     configs = {'save_path': r'C:\Users\Cremecho\Desktop\program\Python38\ML_study_platform\results\multiclass_lr\2024-07-01 10-00-20',
#                'model_name':'multiclass_lr',
#                'epoch':366,
#                'lr': 0.001,
#                'batch_size':100,
#                'val':True}
#     plot_loss(configs)
#     plot_metrics(configs, 'train.txt', [])
#     plot_metrics(configs, 'val.txt', [])