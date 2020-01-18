# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-09 10:06:59
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-20 21:26:08
import os
import random
import sys; sys.path.append('../')
import torch
import torch.nn as nn
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import math
import cv2
from data_loader import TrainDataLoader
from PIL import Image, ImageOps, ImageStat, ImageDraw
from net import SiameseRPN
from torch.nn import init
from shapely.geometry import Polygon

parser = argparse.ArgumentParser(description='PyTorch SiameseRPN Training')

parser.add_argument('--test_path', default='/home/song/srpn/dataset/simple_vot13', metavar='DIR',help='path to dataset')

parser.add_argument('--weight_dir', default='/home/song/srpn/weight', metavar='DIR',help='path to weight')

parser.add_argument('--checkpoint_path', default='/home/song/srpn/weight/epoch_0060_weights.pth.tar', help='resume')

parser.add_argument('--max_epoches', default=100, type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('--max_batches', default=500, type=int, metavar='N', help='number of batch in one epoch')

parser.add_argument('--init_type',  default='xavier', type=str, metavar='INIT', help='init net')

parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='momentum', help='momentum')

parser.add_argument('--weight_decay', '--wd', default=5e-5, type=float, metavar='W', help='weight decay (default: 1e-4)')

def main():
    """ dataloader """
    args = parser.parse_args()
    data_loader = TrainDataLoader(args.test_path, out_feature=25, check = False)

    """ Model on gpu """
    model = SiameseRPN()
    model = model.cuda()
    cudnn.benchmark = True

    """ loss and optimizer """
    criterion = MultiBoxLoss()

    """ load weights """
    init_weights(model)
    if args.checkpoint_path == None:
        sys.exit('please input trained model')
    else:
        assert os.path.isfile(args.checkpoint_path), '{} is not valid checkpoint_path'.format(args.checkpoint_path)
        checkpoint = torch.load(args.checkpoint_path)
        start = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    """ test phase """        
    index_list = range(data_loader.__len__())
    threshold = 50
    precision = []
    precision_c = []
    average_error = []
    average_error_c = []
    iou = []
    iou_c = []
    for example in range(args.max_batches):
        ret = data_loader.__get__(random.choice(index_list)) 
        template = ret['template_tensor'].cuda()
        detection= ret['detection_tensor'].cuda()
        pos_neg_diff = ret['pos_neg_diff_tensor'].cuda()
        cout, rout = model(template, detection) #[1, 10, 17, 17], [1, 20, 17, 17]
        template_img = ret['template_cropped_transformed']
        detection_img = ret['detection_cropped_transformed']

        cout = cout.reshape(-1, 2)
        rout = rout.reshape(-1, 4)
        cout = cout.cpu().detach().numpy()
        score = 1/(1 + np.exp(cout[:,0]-cout[:,1]))
        diff   = rout.cpu().detach().numpy() #1445
        
        num_proposals = 1
        score_64_index = np.argsort(score)[::-1][:num_proposals]

        score64 = score[score_64_index]
        diffs64 = diff[score_64_index, :] 
        anchors64 = ret['anchors'][score_64_index]
        proposals_x = (anchors64[:, 0] + anchors64[:, 2] * diffs64[:, 0]).reshape(-1, 1)
        proposals_y = (anchors64[:, 1] + anchors64[:, 3] * diffs64[:, 1]).reshape(-1, 1)
        proposals_w = (anchors64[:, 2] * np.exp(diffs64[:, 2])).reshape(-1, 1)
        proposals_h = (anchors64[:, 3] * np.exp(diffs64[:, 3])).reshape(-1, 1)
        proposals = np.hstack((proposals_x, proposals_y, proposals_w, proposals_h))

        d = os.path.join(ret['tmp_dir'], '6_pred_proposals')
        if not os.path.exists(d):
            os.makedirs(d)

        template = ret['template_cropped_transformed']
        save_path = os.path.join(ret['tmp_dir'], '6_pred_proposals', '{:04d}_0_template.jpg'.format(example))
        template.save(save_path)

        """traditional correlation match method"""
        template_img = cv2.cvtColor(np.asarray(template_img), cv2.COLOR_RGB2BGR)
        detection_img = cv2.cvtColor(np.asarray(detection_img), cv2.COLOR_RGB2BGR)
        res = cv2.matchTemplate(detection_img, template_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        x1_c = max_loc[0]
        y1_c = max_loc[1]
        """ visualization """
        ratio = ret['detection_cropped_resized_ratio']
        original = Image.open(ret['detection_img_path'])
        origin_w, origin_h = original.size
        x_, y_ = ret['detection_tlcords_of_original_image']
        draw = ImageDraw.Draw(original)
        for i in range(num_proposals):
            x, y, w, h = proposals_x[i], proposals_y[i], proposals_w[i], proposals_h[i]
            x1, y1, x3, y3 = x-w//2, y-h//2, x+w//2, y+h//2

            """ un resized """
            x1, y1, x3, y3 = x1 / ratio, y1 / ratio, x3 / ratio, y3 / ratio
            x1_c, y1_c = x1_c / ratio, y1_c / ratio

            """ un cropped """
            x1_g, y1_g, w, h = ret['template_target_x1y1wh']
            x3_g = x1_g + w
            y3_g = y1_g + h

            x1 = np.clip(x_ + x1, 0, origin_w - 1).astype(np.int32)  # uncropped #target_of_original_img
            y1 = np.clip(y_ + y1, 0, origin_h - 1).astype(np.int32)
            x3 = np.clip(x_ + x3, 0, origin_w - 1).astype(np.int32)
            y3 = np.clip(y_ + y3, 0, origin_h - 1).astype(np.int32)

            x1_c = np.clip(x_ + x1_c, 0, origin_w - 1).astype(np.int32)
            y1_c = np.clip(y_ + y1_c, 0, origin_h - 1).astype(np.int32)
            x3_c = x1_c + ret['template_target_xywh'][2]
            y3_c = y1_c + ret['template_target_xywh'][3]

            draw.line([(x1, y1), (x3, y1), (x3, y3), (x1, y3), (x1, y1)], width=3, fill='yellow')
            draw.line([(x1_g, y1_g), (x3_g, y1_g), (x3_g, y3_g), (x1_g, y3_g), (x1_g, y1_g)], width=3, fill='blue')
            draw.line([(x1_c, y1_c), (x3_c, y1_c), (x3_c, y3_c), (x1_c, y3_c), (x1_c, y1_c)], width=3, fill='red')

        save_path = os.path.join(ret['tmp_dir'], '6_pred_proposals', '{:04d}_1_restore.jpg'.format(example))
        original.save(save_path)
        print('save at {}'.format(save_path))

        """compute iou"""
        s1 = np.array([x1, y1, x3, y1, x3, y3, x1, y3, x1, y1])
        s2 = np.array([x1_g, y1_g, x3_g, y1_g, x3_g, y3_g, x1_g, y3_g, x1_g, y1_g])
        s3 = np.array([x1_c, y1_c, x3_c, y1_c, x3_c, y3_c, x1_c, y3_c, x1_c, y1_c])
        iou.append(intersection(s1, s2))
        iou_c.append(intersection(s3, s2))

        """compute average error"""
        cx = (x1 + x3)/2
        cy = (y1 + y3)/2
        cx_g = (x1_g + x3_g)/2
        cy_g = (y1_g + y3_g) / 2
        cx_c = (x1_c + x3_c)/2
        cy_c = (y1_c + y3_c) / 2
        error = math.sqrt(math.pow(cx-cx_g, 2) + math.pow(cy-cy_g, 2))
        error_c = math.sqrt(math.pow(cx - cx_c, 2) + math.pow(cy - cy_c, 2))
        average_error.append(error)
        average_error_c.append(error_c)
        if error <= threshold:
            precision.append(1)
        else:
            precision.append(0)
        if error_c <= threshold:
            precision_c.append(1)
        else:
            precision_c.append(0)

    iou_mean = np.mean(np.array(iou))
    error_mean = np.mean(np.array(average_error))
    iou_mean_c = np.mean(np.array(iou_c))
    error_mean_c = np.mean(np.array(average_error_c))
    precision = np.mean(np.array(precision))
    precision_c = np.mean(np.array(precision_c))
    print('average iou: {:.4f}'.format(iou_mean))
    print('average error: {:.4f}'.format(error_mean))
    print('average iou for traditional method: {:.4f}'.format(iou_mean_c))
    print('average error for traditional method: {:.4f}'.format(error_mean_c))
    print('precision: {:.4f} @ threshold {:02d}'.format(precision, threshold))
    print('precision for traditional method: {:.4f} @ threshold {:02d}'.format(precision_c, threshold))



def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        # this will apply to each layer
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('conv')!=-1 or classname.find('Linear')!=-1):
            if init_type=='normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#good for relu
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    #print('initialize network with %s' % init_type)
    net.apply(init_func)



class MultiBoxLoss(nn.Module):
    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.closs = torch.nn.CrossEntropyLoss()
        self.rloss = torch.nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        cout, rout = predictions
        cout = cout.reshape(1, 2, -1)
        rout = rout.reshape(-1, 4)
        class_gt, diff = targets[:,0].unsqueeze(0).long(), targets[:,1:]
        closs = self.closs(cout, class_gt)#1,2,*  1,*

        pos_index = np.where(class_gt == 1)[1]
        if pos_index.shape[0] == 0:
            rloss = torch.FloatTensor([0]).cuda()
        else:
            rout_pos = rout[pos_index]
            diff_pos = diff[pos_index]
            
            #print(rout_pos)
            #print(diff_pos)
            rloss = self.rloss(rout_pos, diff_pos) #16
        return closs/64, rloss/16 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(lr, optimizer, epoch, gamma=0.1):
    """Sets the learning rate to the initial LR decayed 0.9 every 50 epochs"""
    lr = lr * (0.9 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def intersection(g, p):
    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter/union

if __name__ == '__main__':
    main()
 

