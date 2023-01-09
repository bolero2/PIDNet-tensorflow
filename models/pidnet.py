import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import BatchNormalization as BatchNorm
import tensorflow.nn as nn

import time
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
import logging


class PIDNet(tf.Module):
    def __init__(self, m=2, n=3, img_size=[1024, 2048], num_classes=19, planes=64, ppm_planes=96, head_planes=128, augment=True):
        super(PIDNet, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes

        self.augment = augment
        self.planes = planes
        self.ppm_planes = ppm_planes
        self.head_planes = head_planes

        self.bn_mom = 0.1
        self.algc = False

        self.relu = ReLU()

        # I branch
        self.conv1 = Sequential([
            Conv2D(self.planes, kernel_size=3, strides=2, padding='same'),
            BatchNorm(momentum=self.bn_mom),
            self.relu,
            Conv2D(self.planes, kernel_size=3, strides=2, padding='same'),
            BatchNorm(momentum=self.bn_mom),
            self.relu,
        ])

        self.layer1 = self._make_layer(BasicBlock, self.planes, self.planes, m)
        self.layer2 = self._make_layer(BasicBlock, self.planes, self.planes * 2, m, strides=2)
        self.layer3 = self._make_layer(BasicBlock, self.planes * 2, self.planes * 4, n, strides=2)
        self.layer4 = self._make_layer(BasicBlock, self.planes * 4, self.planes * 8, n, strides=2)
        self.layer5 = self._make_layer(Bottleneck, self.planes * 8, self.planes * 8, 2, strides=2)

        # P branch
        self.compression3 = Sequential([
            Conv2D(self.planes * 2, kernel_size=1, use_bias=False),
            BatchNorm(momentum=self.bn_mom)
        ])

        self.compression4 = Sequential([
            Conv2D(self.planes * 2, kernel_size=1, use_bias=False),
            BatchNorm(momentum=self.bn_mom)
        ])

        self.pag3 = PagFM(self.planes * 2, self.planes)
        self.pag4 = PagFM(self.planes * 2, self.planes)

        self.layer3_ = self._make_layer(BasicBlock, self.planes * 2, self.planes * 2, m)
        self.layer4_ = self._make_layer(BasicBlock, self.planes * 2, self.planes * 2, m)
        self.layer5_ = self._make_layer(Bottleneck, self.planes * 2, self.planes * 2, m)

        # D branch
        if m == 2:
            self.layer3_d = self._make_single_layer(BasicBlock, self.planes * 2, self.planes)
            self.layer4_d = self._make_layer(Bottleneck, self.planes, self.planes, 1)
            self.diff3 = Sequential([
                Conv2D(self.planes, kernel_size=3, padding='same', use_bias=False),
                BatchNorm(momentum=self.bn_mom)
            ])
            self.diff4 = Sequential([
                Conv2D(self.planes * 2, kernel_size=3, padding='same', use_bias=False),
                BatchNorm(momentum=self.bn_mom)
            ])
            self.spp = PAPPM(self.planes * 16, self.ppm_planes, self.planes * 4)
            self.dfm = Light_Bag(self.planes * 4, self.planes * 4)

        else:
            self.layer3_d = self._make_single_layer(BasicBlock, self.planes * 2, self.planes * 2)
            self.layer4_d = self._make_single_layer(BasicBlock, self.planes * 2, self.planes * 2)
            self.diff3 = Sequential([
                                    Conv2D(self.planes * 2, kernel_size=3, padding='same', use_bias=False),
                                    BatchNorm(momentum=self.bn_mom),
                                    ])
            self.diff4 = Sequential([
                                    Conv2D(self.planes * 2, kernel_size=3, padding='same', use_bias=False),
                                    BatchNorm(momentum=self.bn_mom),
                                    ])
            self.spp = DAPPM(self.planes * 16, self.ppm_planes, self.planes * 4)
            self.dfm = Bag(self.planes * 4, self.planes * 4)

        self.layer5_d = self._make_layer(Bottleneck, self.planes * 2, self.planes * 2, 1)

        # Prediction head
        if self.augment:
            self.seghead_p = segmenthead(self.planes * 2, self.head_planes, self.num_classes)
            self.seghead_d = segmenthead(self.planes * 2, self.planes, 1)

        self.final_layer = segmenthead(self.planes * 4, self.head_planes, self.num_classes)

    def _make_layer(self, block, inplanes, planes, blocks, strides=1):
        downsample = None

        if strides != 1 or inplanes != planes * block.expansion:
            downsample = Sequential([
                Conv2D(planes * block.expansion, kernel_size=1, strides=strides, use_bias=False),
                BatchNorm(mementum=self.bn_mom)
            ])

        layers = []
        layers.append(block(inplanes, planes, strides, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(block(inplanes, planes, strides=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, strides=1, no_relu=False))

        return Sequential(layers)

    def _make_single_layer(self, block, inplanes, planes, strides=1):
        downsample = None

        if strides != 1 or inplanes != planes * block.expansion:
            downsample = Sequential([
                Conv2D(planes * block.expansion, kernel_size=1, strides=strides, use_bias=False),
                BatchNorm(momentum=self.bn_mom)
            ])

        layer = block(inplanes, planes, strides, downsample, no_relu=True)

        return layer
        
    def __call__(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x = self.conv1(x)
        x = self.relu(self.layer1(x))


if __name__ == "__main__":
    model = PIDNet()

    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)

    y = model(x)