import torch
from torch import nn
from layers import *
from layers import ToyCls


class DSNet3D(nn.Module):
    def __init__(self):
        super(DSNet3D, self).__init__()

        self.preBlock_L = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.preBlock_M = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.preBlock_S = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]

        self.featureNum_forw = [96, 96, 192, 192, 192]

        blocks = []
        blocks.append(PostRes(self.featureNum_forw[0], self.featureNum_forw[1]))
        blocks.append(PostRes(self.featureNum_forw[1], self.featureNum_forw[1]))

        setattr(self, 'forw' + str(1), nn.Sequential(*blocks))

        blocks = []
        blocks.append(PostRes(self.featureNum_forw[1], self.featureNum_forw[2]))
        blocks.append(PostRes(self.featureNum_forw[2], self.featureNum_forw[2]))
        blocks.append(PostRes(self.featureNum_forw[2], self.featureNum_forw[2]))

        setattr(self, 'forw' + str(2), nn.Sequential(*blocks))

        blocks = []
        blocks.append(PostRes(self.featureNum_forw[2], self.featureNum_forw[3]))
        blocks.append(PostRes(self.featureNum_forw[3], self.featureNum_forw[3]))
        blocks.append(PostRes(self.featureNum_forw[3], self.featureNum_forw[3]))

        setattr(self, 'forw' + str(3), nn.Sequential(*blocks))

        blocks = []
        blocks.append(PostRes(self.featureNum_forw[3], self.featureNum_forw[4]))
        blocks.append(PostRes(self.featureNum_forw[4], self.featureNum_forw[4]))
        blocks.append(PostRes(self.featureNum_forw[4], self.featureNum_forw[4]))

        setattr(self, 'forw' + str(4), nn.Sequential(*blocks))

        self.maxpool1_L = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool1_M = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool1_S = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 192 * 5 * 5 * 5, 1024),

            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(1024, 1024),

            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 1),

            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

    def forward(self, x_L, x_M, x_S):
        out_L = self.preBlock_L(x_L)
        out_M = self.preBlock_M(x_M)
        out_S = self.preBlock_S(x_S)

        out_pool_L, _ = self.maxpool1_L(out_L)
        out_pool_M, _ = self.maxpool1_M(out_M)
        out_pool_S, _ = self.maxpool1_S(out_S)

        out1 = self.forw1(torch.cat((out_pool_L, out_pool_M, out_pool_S), 1))

        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)

        out3 = self.forw3(out2)

        out4 = self.forw4(out3)

        out = out4.view(out4.size(0), -1)

        out = self.cls(out)

        return out


def get_model_DSNet3D():
    net = DSNet3D()

    loss = torch.nn.BCELoss()

    return net, loss


if __name__ == '__main__':
    net = DSNet3D()

    input3D = torch.ones((2, 1, 20, 20, 20))

    out = net(input3D, input3D, input3D)

    print('out : ', out)
