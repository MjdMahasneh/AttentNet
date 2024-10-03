import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data_loader import LUNA16Dataset, TestingLUNA16Dataset
from utils import *
import numpy as np
from importlib import import_module
from sklearn.metrics import classification_report, f1_score
import shutil
import argparse
import os
import time
import sys
from visualization_helpers import *
from data_helpers import readCSV
import matplotlib.pyplot as plt

print('torch.cuda.device_count() : ', torch.cuda.device_count())
sys.path.append('../')

SCW_flag = False
if SCW_flag:
    from config import config_SCW as config_training
else:
    from config import config_LCL as config_training
    from torchviz import make_dot

seed = 999
np.random.seed(seed)
torch.manual_seed(0)

parser = argparse.ArgumentParser(description='Lung Nodule Detector')

parser.add_argument('--model', '-m', metavar='MODEL', default='backbone',
                    help='model')

parser.add_argument('--epochs', default=30,
                    type=int, metavar='N', help='number of total epochs to run')

parser.add_argument('-j', '--workers', default=6,
                    help='number of data loading workers (default: 8)')

parser.add_argument('-b', '--batch-size', default=64,
                    type=int, metavar='N', help='mini-batch size (default: 64).')

parser.add_argument('--lr', '--learning-rate', default=0.01,
                    type=float, metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9,
                    type=float, metavar='M', help='momentum')

parser.add_argument('--weight-decay', '--wd', default=1e-4,
                    type=float, metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--save-dir', default='./train_results/',
                    type=str, metavar='SAVE', help='directory to save checkpoint (default: none)')

parser.add_argument('--test-output-dir', default='./testing_results/',
                    type=str, metavar='SAVE', help='directory to save testing predictions csv')

parser.add_argument('--test', default=1,
                    help='1 do test evaluation, 0 not')

parser.add_argument('--k-folds-training', default=0,
                    metavar='N', help='1 for k-folds training, 0 not.')

parser.add_argument('--num-sets', default=10,
                    type=int, metavar='N', help='number of subsets')

parser.add_argument('-i', '--item', action='store', dest='outer_sets',
                    type=str, nargs='+', default=['0'],
                    help='leave out subsets')

parser.add_argument('--cls-th', default=0.5,
                    type=float, metavar='N', help='classifier\'s threshold')

parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')


def main(num_sets, outer_sets=[], k_folds_training=False, force_test=False):
    global args

    pos_path = config_training['pos_path']
    neg_path = config_training['neg_path']

    resize_spacing = config_training['resize_spacing']

    size_L = config_training['size_L']
    size_M = config_training['size_M']
    size_S = config_training['size_S']

    resized_voxel = config_training['resized_voxel']

    args = parser.parse_args()

    if force_test:
        args.test = 1
    print('force test {}'.format(bool(force_test)))

    if k_folds_training:
        assert len(outer_sets) == 1, ('outer_sets must have lenght of exactly 1 when k_folds_training is True')
    print('k-folds training {}'.format(bool(k_folds_training)))
    print('num of sets {}, leave out sets {}'.format(num_sets, outer_sets))

    cls_th = args.cls_th

    bestLoss = np.inf
    best_f1 = 0

    model = import_module(args.model)
    net, loss = model.get_model_DSNet3D()

    print('total learnable params {}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))

    save_dir = args.save_dir

    shuffle_train = False
    cross_validation = True

    torch.cuda.set_device(0)

    ntw_vis_path = os.path.join(save_dir, 'network arch - backward')
    if os.path.isfile(ntw_vis_path):
        os.remove(ntw_vis_path)

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

    if args.test != 1:
        net = DataParallel(net)

    print('training args : {}'.format(args))
    print('model summary : ', '\n', net)

    if args.test == 1:
        print('start test')

        original_subset_directory_base = config_training['original_subset_directory_base']
        resampled_subset_directory_base = config_training['resampled_subset_directory_base']
        candidates = config_training['candidates']
        test_result_dir = args.test_output_dir

        testing_batch_size = 30

        single_set_test = True
        if force_test:
            single_set_test = False

        if single_set_test:
            single_set_ID = 0
            print('testing single set {}'.format(single_set_ID))
            test_single_set(network=net, outer_set=single_set_ID,
                            original_subset_directory_base=original_subset_directory_base,
                            resampled_subset_directory_base=resampled_subset_directory_base,
                            candidates=candidates, output_dir=test_result_dir, ckpt_dir=save_dir,
                            resize_spacing=resize_spacing,
                            size_L=size_L, size_M=size_M, size_S=size_S,
                            resized_voxel=resized_voxel,
                            cls_th=cls_th,
                            batch_size=testing_batch_size)
        else:
            test(network=net, num_sets=num_sets,
                 original_subset_directory_base=original_subset_directory_base,
                 resampled_subset_directory_base=resampled_subset_directory_base,
                 candidates=candidates, output_dir=test_result_dir, ckpt_dir=save_dir,
                 resize_spacing=resize_spacing,
                 size_L=size_L, size_M=size_M, size_S=size_S,
                 resized_voxel=resized_voxel,
                 cls_th=cls_th,
                 batch_size=testing_batch_size)
        return

    dataset = LUNA16Dataset(pos_path=pos_path, neg_path=neg_path, size_L=size_L, size_M=size_M, size_S=size_S,
                            resized_voxel=resized_voxel, outer_sets=outer_sets, num_sets=num_sets, phase='train')
    train_loader = DataLoader(dataset, batch_size=int(args.batch_size / 2), shuffle=shuffle_train,
                              num_workers=args.workers, pin_memory=False)

    dataset = LUNA16Dataset(pos_path=pos_path, neg_path=neg_path, size_L=size_L, size_M=size_M, size_S=size_S,
                            resized_voxel=resized_voxel, outer_sets=outer_sets, num_sets=num_sets, phase='test')
    val_loader = DataLoader(dataset, batch_size=int(40 / 2), shuffle=False, num_workers=args.workers, pin_memory=False)

    MGICNN_beta1 = 0.5
    MGICNN_beta2 = 1 - 1e-3

    optimizer = torch.optim.Adam(net.parameters(), args.lr, betas=(MGICNN_beta1, MGICNN_beta2))

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

    for epoch in range(0, args.epochs + 1):

        train(train_loader, net, loss, epoch, optimizer, get_lr, ntw_vis_path)
        print("finsihed epoch {}".format(epoch))

        if cross_validation:

            vali_loss, f1 = validate(val_loader, net, loss, cls_th)

            if bestLoss > vali_loss or f1 > best_f1:

                state_dict = net.module.state_dict()
                for key in state_dict.keys():
                    state_dict[key] = state_dict[key].cpu()

                if k_folds_training:
                    ckpt_path = save_dir + 'subset' + str(outer_sets[0]) + '/'
                else:
                    ckpt_path = save_dir
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)

                detector_name = 'detector_%03d.ckpt' % epoch
                torch.save({
                    'epoch': epoch + 1,
                    'save_dir': ckpt_path,
                    'state_dict': state_dict,
                    'args': args},
                    os.path.join(ckpt_path, detector_name))

                if k_folds_training:
                    shutil.copyfile(os.path.join(ckpt_path, detector_name), os.path.join(ckpt_path, 'weights.ckpt'))

                print("save model on epoch %d" % epoch)
            if bestLoss > vali_loss:
                bestLoss = vali_loss
            if f1 > best_f1:
                best_f1 = f1


def shuffle_in_unison(a, b, c, d):
    assert len(a) == len(b) == len(c) == len(d)

    idxs = np.arange(len(a))
    np.random.shuffle(idxs)

    return a[idxs], b[idxs], c[idxs], d[idxs]


def train(data_loader, net, loss, epoch, optimizer, get_lr, ntw_vis_path):
    ep_start_time = time.time()

    net.train()

    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    s_time = time.time()

    for i, (
    pos_sample_L, pos_sample_M, pos_sample_S, neg_sample_L, neg_sample_M, neg_sample_S, Y_pos, Y_neg) in enumerate(
            data_loader):
        if i != 0:
            if i % 250 == 0:
                print('time passed : ', time.time() - s_time)
                s_time = time.time()

        data_L = torch.cat([pos_sample_L, neg_sample_L], dim=0)
        data_M = torch.cat([pos_sample_M, neg_sample_M], dim=0)
        data_S = torch.cat([pos_sample_S, neg_sample_S], dim=0)

        target = torch.cat([Y_pos, Y_neg], dim=0)

        data_L, data_M, data_S, target = shuffle_in_unison(data_L, data_M, data_S, target)

        data_L = Variable(data_L.cuda(non_blocking=True))
        data_M = Variable(data_M.cuda(non_blocking=True))
        data_S = Variable(data_S.cuda(non_blocking=True))

        target = Variable(target.cuda(non_blocking=True))

        output = net(data_L, data_M, data_S)

        loss_output = loss(output, target)

        optimizer.zero_grad()

        loss_output.backward()

        optimizer.step()

        loss_output = loss_output.item()
        metrics.append(loss_output)

        del pos_sample_L, pos_sample_M, pos_sample_S, neg_sample_L, neg_sample_M, neg_sample_S, Y_pos, Y_neg
        del data_L, data_M, data_S, target

        print("finished iteration {} with loss {}.".format(i, loss_output))

    ep_end_time = time.time()

    if not SCW_flag:
        if os.path.isfile(ntw_vis_path) == False:
            make_dot(output).render(ntw_vis_path)

    metrics = np.asarray(metrics, np.float32)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('-' * 10, 'Training metrics ', '-' * 10)
    print('loss %2.6f' % (np.mean(metrics[:])))

    print('epoch elapsed time : ', ep_end_time - ep_start_time)


def validate(data_loader, net, loss, cls_th):
    start_time = time.time()

    net.eval()

    metrics = []
    all_targets = []
    all_predictions = []

    for i, (
    pos_sample_L, pos_sample_M, pos_sample_S, neg_sample_L, neg_sample_M, neg_sample_S, Y_pos, Y_neg) in enumerate(
            data_loader):
        data_L = torch.cat([pos_sample_L, neg_sample_L], dim=0)
        data_M = torch.cat([pos_sample_M, neg_sample_M], dim=0)
        data_S = torch.cat([pos_sample_S, neg_sample_S], dim=0)

        target = torch.cat([Y_pos, Y_neg], dim=0)

        data_L = Variable(data_L.cuda(non_blocking=True))
        data_M = Variable(data_M.cuda(non_blocking=True))
        data_S = Variable(data_S.cuda(non_blocking=True))

        target = Variable(target.cuda(non_blocking=True))
        all_targets.extend(torch.empty_like(target).copy_(target).detach().cpu().numpy())

        output = net(data_L, data_M, data_S)
        all_predictions.extend(torch.empty_like(output).copy_(output).detach().cpu().numpy())

        loss_output = loss(output, target)

        loss_output = loss_output.item()
        metrics.append(loss_output)

    del pos_sample_L, pos_sample_M, pos_sample_S, neg_sample_L, neg_sample_M, neg_sample_S, Y_pos, Y_neg
    del data_L, data_M, data_S

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)

    print('-' * 10, 'Validation metrics ', '-' * 10)
    print('validation loss %2.6f' % (np.mean(metrics[:])))
    print('validation elapsed time : ', end_time - start_time)

    all_predictions = np.array(all_predictions)
    all_predictions[all_predictions >= cls_th] = 1
    all_predictions[all_predictions < cls_th] = 0

    print('*' * 50, ' classification report : ')
    print(classification_report(all_targets, all_predictions, target_names=['Non-nodule', 'Nodule']))
    print('*' * 50)

    return np.mean(metrics[:]), f1_score(all_targets, all_predictions, average='macro')


def test(network, num_sets,
         original_subset_directory_base, resampled_subset_directory_base, candidates,
         output_dir, ckpt_dir,
         resize_spacing,
         size_L, size_M, size_S,
         resized_voxel,
         cls_th,
         batch_size=128):
    start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    candidates_list = readCSV(candidates)

    output_file = []
    output_file.append('seriesuid,coordX,coordY,coordZ,probability')
    for outer_set in range(num_sets):

        net = network
        ckpt_path = ckpt_dir + 'subset' + str(outer_set) + '/' + 'weights.ckpt'
        print('loading checkpoint {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['state_dict'])

        net = DataParallel(net)
        net.eval()

        print('initializing dataloader of subset {}'.format(outer_set))

        dataset = TestingLUNA16Dataset(original_subset_directory_base=original_subset_directory_base,
                                       resampled_subset_directory_base=resampled_subset_directory_base,
                                       resize_spacing=resize_spacing,
                                       size_L=size_L, size_M=size_M, size_S=size_S,
                                       resized_voxel=resized_voxel,
                                       candidates_list=candidates_list, outer_set=outer_set)

        data_loader_batch_size = 1
        assert data_loader_batch_size == 1
        test_loader = DataLoader(dataset, batch_size=data_loader_batch_size, shuffle=False, num_workers=args.workers,
                                 pin_memory=False)

        for i, (test_patch_list_L, test_patch_list_M, test_patch_list_S, targets) in enumerate(test_loader):

            predictions = []

            test_patch_list_L, test_patch_list_M, test_patch_list_S = test_patch_list_L.squeeze(
                0), test_patch_list_M.squeeze(0), test_patch_list_S.squeeze(0)

            for i in range(0, len(test_patch_list_L), batch_size):
                batch_L = test_patch_list_L[i: i + batch_size]
                batch_M = test_patch_list_M[i: i + batch_size]
                batch_S = test_patch_list_S[i: i + batch_size]

                input_L = Variable(batch_L.cuda(non_blocking=True))
                input_M = Variable(batch_M.cuda(non_blocking=True))
                input_S = Variable(batch_S.cuda(non_blocking=True))

                output = net(input_L, input_M, input_S)

                predictions.extend(output.data.cpu().numpy())

                del input_L, input_M, input_S, output

            print('recording predictions...')
            for i in range(len(targets)):

                if predictions[i][0] >= cls_th:
                    line = targets[i][0][0] + ',' + targets[i][1][0] + ',' + targets[i][2][0] + ',' + targets[i][3][
                        0] + ',' + str(predictions[i][0])
                    output_file.append(line)
        print('subset {} finished'.format(str(outer_set)))

    with open(output_dir + 'METU_VISION_FPRED.csv', 'w') as myFile:
        for item in output_file:
            myFile.write('%s\n' % item)

    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


def test_single_set(network, outer_set,
                    original_subset_directory_base, resampled_subset_directory_base, candidates,
                    output_dir, ckpt_dir,
                    resize_spacing,
                    size_L, size_M, size_S,
                    resized_voxel,
                    cls_th,
                    batch_size=128):
    start_time = time.time()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    candidates_list = readCSV(candidates)

    output_file = []
    output_file.append('seriesuid,coordX,coordY,coordZ,probability')

    net = network
    ckpt_path = ckpt_dir + 'subset' + str(outer_set) + '/' + 'weights.ckpt'
    print('loading checkpoint {}'.format(ckpt_path))
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['state_dict'])

    net = DataParallel(net)
    net.eval()

    print('initializing dataloader of subset {}'.format(outer_set))
    dataset = TestingLUNA16Dataset(original_subset_directory_base=original_subset_directory_base,
                                   resampled_subset_directory_base=resampled_subset_directory_base,
                                   resize_spacing=resize_spacing,
                                   size_L=size_L, size_M=size_M, size_S=size_S,
                                   resized_voxel=resized_voxel,
                                   candidates_list=candidates_list, outer_set=outer_set)
    data_loader_batch_size = 1
    assert data_loader_batch_size == 1
    test_loader = DataLoader(dataset, batch_size=data_loader_batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=False)

    for i, (test_patch_list_L, test_patch_list_M, test_patch_list_S, targets) in enumerate(test_loader):

        predictions = []

        test_patch_list_L, test_patch_list_M, test_patch_list_S = test_patch_list_L.squeeze(
            0), test_patch_list_M.squeeze(0), test_patch_list_S.squeeze(0)

        for i in range(0, len(test_patch_list_L), batch_size):
            batch_L = test_patch_list_L[i: i + batch_size]
            batch_M = test_patch_list_M[i: i + batch_size]
            batch_S = test_patch_list_S[i: i + batch_size]

            input_L = Variable(batch_L.cuda(non_blocking=True))
            input_M = Variable(batch_M.cuda(non_blocking=True))
            input_S = Variable(batch_S.cuda(non_blocking=True))

            output = net(input_L, input_M, input_S)

            predictions.extend(output.data.cpu().numpy())

            del input_L, input_M, input_S, output

        print('recording predictions...')
        for i in range(len(targets)):

            if predictions[i][0] >= cls_th:
                line = targets[i][0][0] + ',' + targets[i][1][0] + ',' + targets[i][2][0] + ',' + targets[i][3][
                    0] + ',' + str(predictions[i][0])
                output_file.append(line)
    print('subset {} finished'.format(str(outer_set)))

    with open(output_dir + 'subset' + str(outer_set) + '_' + 'METU_VISION_FPRED.csv', 'w') as myFile:
        for item in output_file:
            myFile.write('%s\n' % item)

    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))


if __name__ == '__main__':

    args = parser.parse_args()

    if args.test:
        main(num_sets=10, outer_sets=[])
    else:
        if args.k_folds_training:

            assert args.num_sets == 10
            for outer_set in range(args.num_sets):
                print('k_folds_training:{} num_sets:{} outer_sets:{}'.format(args.k_folds_training, args.num_sets,
                                                                             outer_set))
                main(num_sets=args.num_sets, outer_sets=[str(outer_set)], k_folds_training=args.k_folds_training)
            main(num_sets=args.num_sets, outer_sets=[], force_test=True)
        else:

            for s in args.outer_sets:
                assert int(s) < args.num_sets

            main(num_sets=args.num_sets, outer_sets=args.outer_sets, k_folds_training=args.k_folds_training)
