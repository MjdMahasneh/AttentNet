import torch
from torch import nn
from layers import *
import pdb
import math

config = {}
config['anchors'] = [5.0, 10.0, 20.]
config['chanel'] = 1
config['crop_size'] = [128, 128, 128]
config['stride'] = 4
config['max_stride'] = 16
config['num_neg'] = 800
config['th_neg'] = 0.02
config['th_pos_train'] = 0.5
config['th_pos_val'] = 1
config['num_hard'] = 2
config['bound_size'] = 12
config['reso'] = 1
config['sizelim'] = 3.
config['sizelim2'] = 10
config['sizelim3'] = 20
config['aug_scale'] = True
config['r_rand_crop'] = 0.5
config['pad_value'] = 0
config['augtype'] = {'flip': False, 'swap': False, 'scale': False, 'rotate': False}
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38', '990fbe3f0a1b53878669967b9afd1441',
                       'adc3bbc63d40f8761c59be10f1e504c3']
config['dropout'] = 0.3


class Net_2(nn.Module):
    def __init__(self):
        super(Net_2, self).__init__()

        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),

            TanhLeakyRelu(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),

            TanhLeakyRelu()
        )

        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [32, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]

        blocks = []
        blocks.append(ResNeXtLayer(self.featureNum_forw[0], self.featureNum_forw[1]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[1], self.featureNum_forw[1]))

        setattr(self, 'forw' + str(1), nn.Sequential(*blocks))

        blocks = []
        blocks.append(ResNeXtLayer(self.featureNum_forw[1], self.featureNum_forw[2]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[2], self.featureNum_forw[2]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[2], self.featureNum_forw[2]))

        setattr(self, 'forw' + str(2), nn.Sequential(*blocks))

        blocks = []
        blocks.append(ResNeXtLayer(self.featureNum_forw[2], self.featureNum_forw[3]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[3], self.featureNum_forw[3]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[3], self.featureNum_forw[3]))

        setattr(self, 'forw' + str(3), nn.Sequential(*blocks))

        blocks = []
        blocks.append(ResNeXtLayer(self.featureNum_forw[3], self.featureNum_forw[4]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[4], self.featureNum_forw[4]))
        blocks.append(ResNeXtLayer(self.featureNum_forw[4], self.featureNum_forw[4]))

        setattr(self, 'forw' + str(4), nn.Sequential(*blocks))

        blocks = []
        blocks.append(ResNeXtLayer(self.featureNum_back[1] + self.featureNum_forw[2] + 3,
                                   self.featureNum_back[0]))
        blocks.append(ResNeXtLayer(self.featureNum_back[0], self.featureNum_back[0]))
        blocks.append(ResNeXtLayer(self.featureNum_back[0], self.featureNum_back[0]))

        setattr(self, 'back' + str(2), nn.Sequential(*blocks))

        blocks = []
        blocks.append(ResNeXtLayer(self.featureNum_back[2] + self.featureNum_forw[3] + 0,
                                   self.featureNum_back[1]))
        blocks.append(ResNeXtLayer(self.featureNum_back[1], self.featureNum_back[1]))
        blocks.append(ResNeXtLayer(self.featureNum_back[1], self.featureNum_back[1]))

        setattr(self, 'back' + str(3), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),

            TanhLeakyRelu()
        )
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),

            TanhLeakyRelu()
        )

        self.drop1 = nn.Dropout3d(p=config['dropout'], inplace=False)
        self.drop2 = nn.Dropout3d(p=config['dropout'], inplace=False)
        self.drop3 = nn.Dropout3d(p=config['dropout'], inplace=False)
        self.drop4 = nn.Dropout3d(p=config['dropout'], inplace=False)

        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),

                                    TanhLeakyRelu(),

                                    nn.Conv3d(64, 5 * len(config['anchors']), kernel_size=1))

        self.nodule_output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),

                                           TanhLeakyRelu(),

                                           nn.Conv3d(64, len(config['anchors']), kernel_size=1))

        self.regress_output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),

                                            TanhLeakyRelu(),

                                            nn.Conv3d(64, 4 * len(config['anchors']), kernel_size=1))
        focal_bias = -math.log((1.0 - 0.01) / 0.01)
        self._modules['nodule_output'][2].bias.data.fill_(focal_bias)

    def forward(self, x, coord):
        out = self.preBlock(x)
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)
        out1 = self.drop1(out1)

        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)
        out2 = self.drop2(out2)

        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)
        out3 = self.drop3(out3)

        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))
        rev2 = self.path2(comb3)

        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))
        comb2 = self.drop4(comb2)

        nodule_out = self.nodule_output(comb2)
        regress_out = self.regress_output(comb2)
        nodule_size = nodule_out.size()
        regress_size = regress_out.size()

        nodule_out = nodule_out.view(nodule_out.size(0), nodule_out.size(1), -1)
        regress_out = regress_out.view(regress_out.size(0), regress_out.size(1), -1)

        nodule_out = nodule_out.transpose(1, 2).contiguous().view(nodule_size[0], nodule_size[2], nodule_size[3],
                                                                  nodule_size[4], len(config['anchors']), 1)
        regress_out = regress_out.transpose(1, 2).contiguous().view(regress_size[0], regress_size[2], regress_size[3],
                                                                    regress_size[4], len(config['anchors']), 4)
        out = torch.cat((nodule_out, regress_out), 5)

        return out


def get_model_2():
    net = Net_2()
    loss = FocalLoss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
