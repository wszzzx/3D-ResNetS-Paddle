import math
from functools import partial
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv3D
import numpy as np
import paddle.fluid as fluid
from collections import OrderedDict

def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return Conv3D(in_planes,
                  out_planes,
                  filter_size=3,
                  stride=stride,
                  padding=1,
                  bias_attr=False, param_attr=fluid.initializer.MSRAInitializer(uniform=False, fan_in=None, seed=10))


def conv1x1x1(in_planes, out_planes, stride=1):
    return Conv3D(in_planes,
                  out_planes,
                  filter_size=1,
                  stride=stride,
                  bias_attr=False, param_attr=fluid.initializer.MSRAInitializer(uniform=False, fan_in=None, seed=10))


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = fluid.dygraph.BatchNorm(num_channels = planes,momentum=0.9, epsilon=1e-05, data_layout='NCHW')
        # self.bn1 = fluid.BatchNorm(input=planes, momentum=0.9, epsilon=1e-05, data_layout='NCHW')
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = fluid.dygraph.BatchNorm(planes,momentum=0.9, epsilon=1e-05, data_layout='NCHW')
        # self.bn2 = fluid.layers.batch_norm(input=planes,momentum=0.9,epsilon=1e-05,data_layout='NCHW')
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(input=out)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        out = self.bn2(input=out,)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


class Bottleneck(fluid.dygraph.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = fluid.dygraph.BatchNorm(planes,momentum=0.9, epsilon=1e-05, data_layout='NCHW')
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = fluid.dygraph.BatchNorm(planes,momentum=0.9, epsilon=1e-05, data_layout='NCHW')
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = fluid.dygraph.BatchNorm(planes * self.expansion,momentum=0.9, epsilon=1e-05, data_layout='NCHW')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(input=out,)
        out = fluid.layers.relu(out)

        out = self.conv2(out)
        out = self.bn2(input=out,)
        out = fluid.layers.relu(out)

        out = self.conv3(out)
        out = self.bn3(input=out,)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = fluid.layers.relu(out)

        return out


# 34 model = ResNet(BasicBlock, [3, 4, 6, 3], [64, 128, 256, 512], **kwargs)
class ResNet(fluid.dygraph.Layer):

    def __init__(self,
                 block,  # BasicBlock
                 layers,  # [3, 4, 6, 3]
                 block_inplanes,  # [64, 128, 256, 512]
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=101):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]  # [64, 128, 256, 512]

        self.in_planes = np.array(block_inplanes[0])  # 64
        self.no_max_pool = no_max_pool

        self.conv1 = fluid.dygraph.nn.Conv3D(n_input_channels,
                                             self.in_planes,  # 64
                                             filter_size=(conv1_t_size, 7, 7),
                                             stride=(conv1_t_stride, 2, 2),
                                             padding=(conv1_t_size // 2, 3, 3),
                                             bias_attr=False,
                                             param_attr=fluid.initializer.MSRAInitializer(uniform=False, fan_in=None,
                                                                                          seed=10)
                                             )

        self.bn1 = fluid.dygraph.BatchNorm(self.in_planes,momentum=0.9,epsilon=1e-05,data_layout='NCHW')
        # self.relu = fluid.layers.relu()
        # self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = fluid.layers.pool3d(pool_type ='max',pool_stride  =3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        # self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.avgpool = fluid.layers.adaptive_pool3d((1, 1, 1))

        # self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        self.fc = fluid.dygraph.Linear(block_inplanes[3] * block.expansion, n_classes, act='softmax')

        # for m in self.modules():
        #     # if isinstance(m, nn.Conv3d):
        #     if isinstance(m, fluid.dygraph.nn.Conv3D):
        #
        #         # nn.init.kaiming_normal_(m.weight,
        #         #                         mode='fan_out',
        #         #                         nonlinearity='relu')
        #         # kaiming_normal(m.weight,
        #         #                         mode='fan_out',
        #         #                         nonlinearity='relu')
        #         fluid.initializer.MSRAInitializer(uniform=False,fan_in = None,seed = 10)
        #     elif isinstance(m, fluid.layers.batch_norm):
        #         # nn.init.constant_(m.weight, 1)
        #         # nn.init.constant_(m.bias, 0)
        #         fluid.initializer.ConstantInitializer(m.weight, 1)
        #         fluid.initializer.ConstantInitializer(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        # out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        # zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
        #                         out.size(3), out.size(4))
        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()
        #
        # out = torch.cat([out.data, zero_pads], dim=1)

        out = fluid.layers.pool3d(x, filter_size=1, stride=stride, pool_type='avg')

        zero_pads = fluid.layers.zeros(out.size(0), planes - out.size(1), out.size(2),
                                       out.size(3), out.size(4))

        # if isinstance(out.data, torch.cuda.FloatTensor):
        #     zero_pads = zero_pads.cuda()

        # out = torch.cat([out.data, zero_pads], dim=1)
        out = fluid.layers.concat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                # downsample = fluid.dygraph.Sequential(
                #     conv1x1x1(self.in_planes, planes * block.expansion, stride),
                #     nn.BatchNorm3d(planes * block.expansion),)
                downsample = fluid.dygraph.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    fluid.dygraph.BatchNorm(num_channels=planes * block.expansion, momentum=0.9, epsilon=1e-05,
                                    data_layout='NCHW'), )

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return fluid.dygraph.Sequential(*layers)

    def forward(self, x, label=None):
        x = self.conv1(x)
        # print("x1",x)
        x = self.bn1(input=x, )
        # x = self.bn1(x)
        x = fluid.layers.relu(x)
        # x = self.relu(x)
        if not self.no_max_pool:
            x = fluid.layers.pool3d(input=x, pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)
        # print('x',x)
        x = self.layer1(x)
        # print('layer1',x.shape)
        x = self.layer2(x)
        # print('layer2',x.shape)
        x = self.layer3(x)
        # print('layer3',x.shape)
        x = self.layer4(x)
        # print('layer4',x.shape)

        x = fluid.layers.adaptive_pool3d(input=x, pool_size=(1, 1, 1),pool_type='avg')

        # x = x.view(x.size(0), -1)
        x = fluid.layers.reshape(x, shape=(x.shape[0], -1))

        x = self.fc(x)

        if label is not None:
            acc = fluid.layers.accuracy(input=x, label=label)
            return x, acc
        else:
            return x



def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

