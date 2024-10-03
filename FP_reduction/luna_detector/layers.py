import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

config = {}
config['hidden_size'] = 512
config['num_layers'] = 4
config['image_size'] = (20, 20)
config['dropout_rate'] = 0.1
config['attention_dropout_rate'] = 0.0
config['mlp_dim'] = 1024
config['num_heads'] = 8

ACT2FN = {"gelu": torch.nn.functional.gelu}


class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

        self.num_attention_heads = config['num_heads']

        self.attention_head_size = int(config['hidden_size'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config['hidden_size'], self.all_head_size)
        self.key = nn.Linear(config['hidden_size'], self.all_head_size)
        self.value = nn.Linear(config['hidden_size'], self.all_head_size)

        self.out = nn.Linear(config['hidden_size'], config['hidden_size'])
        self.attn_dropout = nn.Dropout(config['attention_dropout_rate'])
        self.proj_dropout = nn.Dropout(config['attention_dropout_rate'])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


class Mlp(nn.Module):

    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config['hidden_size'], config['mlp_dim'])
        self.fc2 = nn.Linear(config['mlp_dim'], config['hidden_size'])
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config['dropout_rate'])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, img_size=config['image_size'], dropout_rate=config['dropout_rate'], in_channels=192):
        super(Embeddings, self).__init__()

        """ TransUNet
        grid_size = 4, 16, 16 
        img_size = 64,256,256
        patch_size = 1, 1, 1
        patch_size_real = 16, 16, 16 
        n_patches = 1024
        """

        """ LUNA-16 DS-detector:
        grid_size = 16, 16, 16 
        img_size = 128,128,128
        patch_size = 1, 1, 1
        patch_size_real = 16, 16, 16 
        n_patches = 512
        """

        patch_size = (2, 2)

        patch_size_real = (patch_size[0] * 1, patch_size[1] * 1)
        n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,

                                          out_channels=config['hidden_size'],
                                          kernel_size=patch_size,
                                          stride=patch_size)

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config['hidden_size']))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.patch_embeddings(x)

        x = x.flatten(2)

        x = x.transpose(-1, -2)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class Block(nn.Module):

    def __init__(self):
        super(Block, self).__init__()

        self.attention_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)

        self.ffn = Mlp()

        self.attn = Attention()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)

        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h

        return x


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.layer = nn.ModuleList()

        self.encoder_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        for _ in range(config['num_layers']):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):

        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)

        return encoded


class Get3D(nn.Module):
    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()

        ch, h, w = config['image_size']
        ch, h, w = int(ch / 2), int(h / 2), int(w / 2)

        x = hidden_states.permute(0, 2, 1)

        x = x.contiguous().view(B, hidden, ch, h, w)
        return x


class Get2D(nn.Module):
    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()

        h, w = config['image_size']
        h, w = int(h / 2), int(w / 2)

        x = hidden_states.permute(0, 2, 1)

        x = x.contiguous().view(B, hidden, h, w)

        return x


class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        self.embeddings = Embeddings(in_channels=192)

        self.encoder = Encoder()

        self.get2d = Get2D()

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded = self.encoder(embedding_output)

        encoded_2d = self.get2d(encoded)

        return encoded_2d


class ReductionBlock(nn.Module):
    def __init__(self, latent_channels):
        super(ReductionBlock, self).__init__()

        t_reduction = config['hidden_size'] // latent_channels

        self.conv1 = nn.Conv3d(config['hidden_size'], config['hidden_size'] // t_reduction, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(config['hidden_size'] // t_reduction)

        self.conv2 = nn.Conv3d(config['hidden_size'] // t_reduction, config['hidden_size'] // t_reduction,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(config['hidden_size'] // t_reduction)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)

        return out


class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(n_in, n_out, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ToyCls(nn.Module):
    def __init__(self, n_in):
        super(ToyCls, self).__init__()

        self.cls = nn.Sequential(

            nn.Linear(1 * n_in * 3 * 10 * 10, 256),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.cls(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelAttentionGate(nn.Module):

    def __init__(self, channel, reduction=16):
        super(ChannelAttentionGate, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.mlp = nn.Sequential(

            Flatten(),
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()

        y_avg = self.avg_pool(x).view(b, c)

        y_avg = self.mlp(y_avg).view(b, c, 1, 1, 1)

        y_max = self.max_pool(x).view(b, c)

        y_max = self.mlp(y_max).view(b, c, 1, 1, 1)

        channel_att_sum = y_avg + y_max

        scale = self.sigmoid(channel_att_sum)

        return x * scale


class SpatialConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(SpatialConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        self.bn = nn.BatchNorm3d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttentionGate(nn.Module):
    def __init__(self):
        super(SpatialAttentionGate, self).__init__()
        kernel_size = 7

        self.spatial_conv_block = SpatialConvBlock(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                                                   relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_pools = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

        x_out = self.spatial_conv_block(channel_pools)
        scale = self.sigmoid(x_out)
        return x * scale


class CBAM(nn.Module):

    def __init__(self, gate_channels, reduction_ratio=16, no_spatial=False):
        super(CBAM, self).__init__()
        self.no_spatial = no_spatial
        self.channel_attention_gate = ChannelAttentionGate(gate_channels, reduction_ratio)
        if not no_spatial:
            self.spatial_attention_gate = SpatialAttentionGate()

    def forward(self, x):
        x_out = self.channel_attention_gate(x)
        if not self.no_spatial:
            x_out = self.spatial_attention_gate(x_out)
        return x_out


class TanhLeakyRelu(nn.Module):
    def __init__(self):
        super(TanhLeakyRelu, self).__init__()

    def forward(self, x):
        x = torch.maximum(torch.tanh(x), x)
        return x


class ZoomConv(nn.Module):
    def __init__(self, n_in, n_out):
        super(ZoomConv, self).__init__()

        self.conv1 = nn.ConvTranspose3d(n_in, n_out, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm3d(n_out)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pool(out)

        return out


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)

        self.cbam = CBAM(n_out, reduction_ratio=2, no_spatial=True)

        self.zoom = ZoomConv(n_in, n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):

        zoom = self.zoom(x)

        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cbam(out)

        out += residual
        out += zoom

        out = self.relu(out)
        return out


def hard_mining_II(predictions, labels, ratio, pos):
    num_inst = predictions.size(0)
    num_hard = max(int(ratio * num_inst), 1)

    if not pos:

        _, idcs = torch.topk(input=predictions, k=min(num_hard, len(predictions)), largest=True)
    else:

        _, idcs = torch.topk(input=predictions, k=min(num_hard, len(predictions)), largest=False)

    predictions = torch.index_select(predictions, 0, idcs)
    labels = torch.index_select(labels, 0, idcs)

    return predictions, labels


class BCEOHEM_pos_class(nn.Module):
    def __init__(self, ratio):
        print('BCE OHEM ratio {}'.format(ratio))
        super(BCEOHEM_pos_class, self).__init__()
        assert ratio <= 1
        self.ratio = ratio

    def forward(self, inputs, targets):

        pos_idcs = targets[:, 0] >= 0.5
        pos_output = inputs[:, 0][pos_idcs]
        pos_labels = targets[:, 0][pos_idcs]

        neg_idcs = targets[:, 0] < 0.5

        neg_output = inputs[neg_idcs]
        neg_labels = targets[neg_idcs]

        if len(pos_output) == 0 and len(neg_output) == 0:
            raise Exception('Can\'t compute loss with empty Tensors. len(pos_output) = 0 and len(neg_output) = 0')
        elif len(pos_output) == 0:

            neg_BCE_loss = F.binary_cross_entropy(neg_output, neg_labels, reduction='none')

            return torch.mean(neg_BCE_loss)
        elif len(neg_output) == 0:

            pos_output, pos_labels = hard_mining_II(pos_output, pos_labels, self.ratio, pos=True)

            pos_BCE_loss = F.binary_cross_entropy(pos_output, pos_labels, reduction='none')

            return torch.mean(pos_BCE_loss)
        else:

            neg_BCE_loss = F.binary_cross_entropy(neg_output, neg_labels, reduction='none')

            pos_output, pos_labels = hard_mining_II(pos_output, pos_labels, self.ratio, pos=True)

            pos_BCE_loss = F.binary_cross_entropy(pos_output, pos_labels, reduction='none')

            classify_loss = 0.5 * pos_BCE_loss + 0.5 * neg_BCE_loss

            return torch.mean(classify_loss)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False):
        super(FocalLoss, self).__init__()
        print("FOCAL LOSS", gamma, alpha)

        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)
