import argparse
import os
import time
import numpy as np

from data_loader_train_by_Joint_cross_section_2 import LungNodule3Ddetector, collate

from importlib import import_module
import shutil
from utils import *
import sys

sys.path.append('../')
from split_combine import SplitComb

import torch

print('torch.cuda.device_count() : ', torch.cuda.device_count())

from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader

from torch.autograd import Variable

SCW_flag = True
if SCW_flag:
    from config_training_SCW import config as config_training
else:
    from config_training import config as config_training

seed = 999

np.random.seed(seed)

parser = argparse.ArgumentParser(description='Lung Nodule Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18_se_unzipped_ResNeXt_3_with_AttendType5_DropOut',
                    help='model')

parser.add_argument('-j', '--workers', default=8,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=250,
                    type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=8,

                    type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--resume', default='',

                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir',
                    default='./train_results/ResNEXt-3_Att-T5_TanHLeakyReLU_JointCrsSectAppraoch_DropOutSurgery/',

                    type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--test', default=0,
                    help='1 do test evaluation, 0 not')
parser.add_argument('--testing-aug', default=0,
                    help='1 perform testing time augmentation, 0 not')

parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')

parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

parser.add_argument('--cross_sect_aug', default=1, type=int, metavar='N',
                    help='1, for cross sectional augmentation.')

parser.add_argument('-i', '--item', action='store', dest='cross_sections',
                    type=str, nargs='+', default=['axial', 'coronal', 'sagittal'],
                    help="cross section/s to be used in training/testing. Takes one or more of axial, coronal, saggital. Examples: -i axial, -i coronal, -i sagittal")

parser.add_argument('--train-set', default='./luna_val-train_20-80_train.npy',
                    type=str, metavar='TRAINSET',
                    help='path to training data set')

parser.add_argument('--test-set', default='./luna_val-train_20-80_val.npy',
                    type=str, metavar='TESTSET',
                    help='path to testing data set')


def main():
    global args
    args = parser.parse_args()

    bestLoss = np.inf

    torch.manual_seed(0)
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model_2()

    start_epoch = args.start_epoch

    save_dir = args.save_dir

    train_set = args.train_set
    test_set = args.test_set

    cross_sect_aug = bool(args.cross_sect_aug)

    cross_sections = args.cross_sections
    print('Cross sections : ', cross_sections)

    testing_aug = args.testing_aug

    randomize_on_wrap_around = False

    shuffle_train = False

    ntw_vis_path = os.path.join(save_dir, 'network arch - backward')
    if os.path.isfile(ntw_vis_path):
        os.remove(ntw_vis_path)

    if args.resume:
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(save_dir + 'detector_' + args.resume)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if
                   f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    loss = loss.cuda()

    cudnn.benchmark = True
    net = DataParallel(net)
    datadir = config_training['preprocess_result_path']

    print('training args : ', '\n', args)
    print('model summary : ', '\n', net)

    luna_train = np.load(train_set)
    luna_test = np.load(test_set)

    if args.test == 1:
        print("start test")
        margin = 32
        sidelen = 144
        test_batch_size = 1

        split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])

        dataset_axi = LungNodule3Ddetector(datadir, luna_test, config, phase='test', split_comber=split_comber,
                                           cross_sect='axial')
        test_loader_axi = DataLoader(dataset_axi, batch_size=test_batch_size, shuffle=False, num_workers=args.workers,
                                     collate_fn=collate, pin_memory=False)

        dataset_cor = LungNodule3Ddetector(datadir, luna_test, config, phase='test', split_comber=split_comber,
                                           cross_sect='coronal')
        test_loader_cor = DataLoader(dataset_cor, batch_size=test_batch_size, shuffle=False, num_workers=args.workers,
                                     collate_fn=collate, pin_memory=False)

        dataset_sag = LungNodule3Ddetector(datadir, luna_test, config, phase='test', split_comber=split_comber,
                                           cross_sect='sagittal')
        test_loader_sag = DataLoader(dataset_sag, batch_size=test_batch_size, shuffle=False, num_workers=args.workers,
                                     collate_fn=collate, pin_memory=False)

        print('testing time augmentation is set to {}'.format(bool(testing_aug)))
        if testing_aug:
            cross_sectiona_aug_test(test_loader_axi, test_loader_cor, test_loader_sag, net, get_pbb, save_dir, config,
                                    cross_sections)
        else:

            test(test_loader_axi, net, get_pbb, save_dir, config)

        return

    dataset_axi = LungNodule3Ddetector(datadir, luna_train, config, phase='train', cross_sect='axial',
                                       randomize_on_wrap_around=randomize_on_wrap_around)

    dataset_cor = LungNodule3Ddetector(datadir, luna_train, config, phase='train', cross_sect='coronal',
                                       randomize_on_wrap_around=randomize_on_wrap_around)

    dataset_sag = LungNodule3Ddetector(datadir, luna_train, config, phase='train', cross_sect='sagittal',
                                       randomize_on_wrap_around=randomize_on_wrap_around)

    train_loader_axi = DataLoader(dataset_axi, batch_size=args.batch_size, shuffle=shuffle_train,
                                  num_workers=args.workers,
                                  pin_memory=True)

    train_loader_cor = DataLoader(dataset_cor, batch_size=args.batch_size, shuffle=shuffle_train,
                                  num_workers=args.workers,
                                  pin_memory=True)

    train_loader_sag = DataLoader(dataset_sag, batch_size=args.batch_size, shuffle=shuffle_train,
                                  num_workers=args.workers,
                                  pin_memory=True)

    dataset_axi = LungNodule3Ddetector(datadir, luna_test, config, phase='val', cross_sect='axial')
    val_loader_axi = DataLoader(dataset_axi, batch_size=2,
                                shuffle=False, num_workers=args.workers, pin_memory=True)

    optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay)

    def get_lr(epoch):
        if epoch <= args.epochs * 0.2:
            lr = args.lr
        elif epoch <= args.epochs * 0.4:
            lr = 0.1 * args.lr
        elif epoch <= args.epochs * 0.6:
            lr = 0.05 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    for epoch in range(start_epoch, args.epochs + 1):

        train(train_loader_axi, train_loader_cor, train_loader_sag, net, loss, epoch, optimizer, get_lr, ntw_vis_path,
              cross_sect_aug)
        print("finsihed epoch {}".format(epoch))

        vali_loss = validate(val_loader_axi, net, loss)

        if bestLoss > vali_loss:
            bestLoss = vali_loss
            state_dict = net.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch + 1,
                'save_dir': save_dir,
                'state_dict': state_dict,
                'args': args},
                os.path.join(save_dir,
                             'detector_%03d.ckpt' % epoch))
            print("save model on epoch %d" % epoch)


def train(data_loader_axi, data_loader_cor, data_loader_sag, net, loss, epoch, optimizer, get_lr, ntw_vis_path,
          cross_sect_aug):
    start_time = time.time()

    net.train()

    lr = get_lr(epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    s_time = time.time()
    for i, (dicts_axi, dicts_cor, dicts_sag) in enumerate(zip(data_loader_axi, data_loader_cor, data_loader_sag)):

        if i != 0:
            if i % 20 == 0:
                print('time passed : ', time.time() - s_time)
                s_time = time.time()

        view = 'axial'

        if cross_sect_aug:
            if i % 3 == 0:

                if np.random.randint(0, 10) % 2 == 0:
                    view = 'coronal'
                else:
                    view = 'sagittal'

        if view == 'axial':
            data, target, coord = dicts_axi
        if view == 'coronal':
            data, target, coord = dicts_cor
        if view == 'sagittal':
            data, target, coord = dicts_sag

        joint_loss_output = []

        data = Variable(data.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        coord = Variable(coord.cuda(non_blocking=True))

        output = net(data, coord)

        loss_output = loss(output, target)

        optimizer.zero_grad()

        loss_output[0].backward()

        optimizer.step()

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

        print("finished {} iteration {} with loss {}.".format(view, i, loss_output[0]))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)

    print('Epoch %03d (lr %.5f)' % (epoch, lr))

    print('-' * 10, 'Training metrics of ', '-' * 10)

    print(' tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))

    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))


def validate(data_loader_axi, net, loss):
    start_time = time.time()

    net.eval()

    metrics = []

    for i, (dicts_axi) in enumerate(data_loader_axi):
        data, target, coord = dicts_axi

        data = Variable(data.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        coord = Variable(coord.cuda(non_blocking=True))

        output = net(data, coord)
        loss_output = loss(output, target, train=False)

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)

    print('-' * 10, 'Validation metrics of ', '-' * 10)

    print('tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))

    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5])))

    return np.mean(metrics[:, 0])


def test(data_loader_axi, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []

    split_comber = data_loader_axi.dataset.split_comber

    for i_name, (dicts_axi) in enumerate(data_loader_axi):

        data, target, coord, nzhw = dicts_axi
        name = data_loader_axi.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]

        pbb_cross_sectional = []

        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]

        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())

        splitlist = list(range(0, len(data) + 1, n_per_run))
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]].cuda(non_blocking=True))
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]].cuda(non_blocking=True))
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
            del input, inputcoord, output
            torch.cuda.empty_cache()

        output = np.concatenate(outputlist, 0)

        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)

        print('i_name, and name : ', [i_name, name])

        np.save(os.path.join(save_dir, name + '_pbb.npy'), np.array(pbb))
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)

    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


def cross_sectiona_aug_test(data_loader_axi, data_loader_cor, data_loader_sag, net, get_pbb, save_dir, config,
                            cross_sections):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []

    if 'axial' in cross_sections:
        split_comber_axi = data_loader_axi.dataset.split_comber
    if 'coronal' in cross_sections:
        split_comber_cor = data_loader_axi.dataset.split_comber
    if 'sagittal' in cross_sections:
        split_comber_sag = data_loader_axi.dataset.split_comber

    print('split_comber_axi split_comber_cor and split_comber_sag')
    print(split_comber_axi)
    print(split_comber_sag)
    print(split_comber_cor)
    print(split_comber_axi == split_comber_sag)
    print(split_comber_axi == split_comber_cor)
    print(split_comber_cor == split_comber_sag)

    temp_count = 0
    for i_name, (dicts_axi, dicts_cor, dicts_sag) in enumerate(zip(data_loader_axi, data_loader_cor, data_loader_sag)):

        data_dict_axi, target_dict_axi, coord_dict_axi, nzhw_dict_axi = dicts_axi
        data_dict_cor, target_dict_cor, coord_dict_cor, nzhw_dict_cor = dicts_cor
        data_dict_sag, target_dict_sag, coord_dict_sag, nzhw_dict_sag = dicts_sag

        pbb_cross_sectional = []
        for cross_section in cross_sections:

            print(' ---- section step counter : ', temp_count % 3 + 1)
            temp_count += 1

            if cross_section == 'axial':
                data, target, coord, nzhw = data_dict_axi, target_dict_axi, coord_dict_axi, nzhw_dict_axi
                name = data_loader_axi.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
            elif cross_section == 'coronal':
                data, target, coord, nzhw = data_dict_cor, target_dict_cor, coord_dict_cor, nzhw_dict_cor
                name = data_loader_cor.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
            elif cross_section == 'sagittal':
                data, target, coord, nzhw = data_dict_sag, target_dict_sag, coord_dict_sag, nzhw_dict_sag
                name = data_loader_sag.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]

            target = [np.asarray(t, np.float32) for t in target]
            lbb = target[0]
            nzhw = nzhw[0]

            data = data[0][0]
            coord = coord[0][0]
            isfeat = False
            if 'output_feature' in config:
                if config['output_feature']:
                    isfeat = True
            n_per_run = args.n_test
            print(data.size())

            splitlist = list(range(0, len(data) + 1, n_per_run))
            if splitlist[-1] != len(data):
                splitlist.append(len(data))
            outputlist = []
            featurelist = []

            for i in range(len(splitlist) - 1):
                input = Variable(data[splitlist[i]:splitlist[i + 1]].cuda(non_blocking=True))
                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]].cuda(non_blocking=True))
                if isfeat:
                    output, feature = net(input, inputcoord)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input, inputcoord)
                outputlist.append(output.data.cpu().numpy())
                del input, inputcoord, output
                torch.cuda.empty_cache()

            output = np.concatenate(outputlist, 0)

            if cross_section == 'axial':
                output = split_comber_axi.combine(output, nzhw=nzhw)
            elif cross_section == 'coronal':
                output = split_comber_cor.combine(output, nzhw=nzhw)
            elif cross_section == 'sagittal':
                output = split_comber_sag.combine(output, nzhw=nzhw)
            if isfeat:
                feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]

                if cross_section == 'axial':
                    feature = split_comber_axi.combine(feature, sidelen)[..., 0]
                elif cross_section == 'coronal':
                    feature = split_comber_cor.combine(feature, sidelen)[..., 0]
                elif cross_section == 'sagittal':
                    feature = split_comber_sag.combine(feature, sidelen)[..., 0]

            thresh = -3
            pbb, mask = get_pbb(output, thresh, ismask=True)
            if isfeat:
                feature_selected = feature[mask[0], mask[1], mask[2]]
                np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)

            print('i_name, and name : ', [i_name, name])

            pbb_prime = np.empty_like(pbb)

            if cross_section == 'axial':
                pbb_prime = pbb

            elif cross_section == 'coronal':

                pbb_prime[:, 3] = pbb[:, 3]
                pbb_prime[:, 2] = pbb[:, 1]
                pbb_prime[:, 1] = pbb[:, 2]
                pbb_prime[:, 4] = pbb[:, 4]

                pbb_prime[:, 0] = pbb[:, 0]
            elif cross_section == 'sagittal':

                pbb_prime[:, 3] = pbb[:, 1]
                pbb_prime[:, 2] = pbb[:, 3]
                pbb_prime[:, 1] = pbb[:, 2]
                pbb_prime[:, 4] = pbb[:, 4]

                pbb_prime[:, 0] = pbb[:, 0]

            pbb_cross_sectional.extend(pbb_prime)

        np.save(os.path.join(save_dir, name + '_pbb.npy'), np.array(pbb_cross_sectional))
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)

    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


if __name__ == '__main__':
    main()
