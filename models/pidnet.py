import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU
from tensorflow.keras.layers import BatchNormalization as BatchNorm
import tensorflow.nn as nn

import time
import logging
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag


class PIDNet(tf.keras.Model):
    def __init__(self, m=2, n=3, img_size=[1024, 2048], num_classes=19, planes=32, ppm_planes=96, head_planes=128, augment=True):
        super().__init__()
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
                BatchNorm(momentum=self.bn_mom)
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
        
    def call(self, x):
        width_output = x.shape[-2] // 8
        height_output = x.shape[-3] // 8

        x = self.conv1(x)
        x = self.relu(self.layer1(x))

        x = self.relu(self.layer2(x))

        x_ = self.layer3_(x)
        x_d = self.layer3_d(x)

        x = self.relu(self.layer3(x))

        x_ = self.pag3(x_, self.compression3(x))
        x_d = x_d + tf.image.resize(
            self.diff3(x),
            size=[height_output, width_output],
            method='bilinear'
        )

        if self.augment:
            temp_p = x_

        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_d = self.layer4_d(self.relu(x_d))

        x_ = self.pag4(x_, self.compression4(x))
        x_d = x_d + tf.image.resize(
            self.diff4(x),
            size=[height_output, width_output],
            method='bilinear'
        )
        
        if self.augment:
            temp_d = x_d

        x_ = self.layer5_(self.relu(x_))
        x_d = self.layer5_d(self.relu(x_d))

        x = tf.image.resize(
            self.spp(self.layer5(x)),
            size=[height_output, width_output],
            method='bilinear'
        )

        x_ = self.final_layer(self.dfm(x_, x, x_d))

        if self.augment:
            x_extra_p = self.seghead_p(temp_p)
            x_extra_d = self.seghead_d(temp_d)
            return [x_extra_p, x_, x_extra_d]
        else:
            return x_

def get_seg_model(cfg, pretrained=""):
    print(f"\n >>> Model name : {cfg.MODEL.NAME}")

    model_name = cfg.MODEL.NAME.lower().replace("-", "_")
    if 'pidnet_s' in model_name:
        print("\n >>> Load PIDNet-Small model")
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=32, ppm_planes=96, head_planes=128, augment=True) 
    elif 'pidnet_m' in model_name:
        print("\n >>> Load PIDNet-Medium model")
        model = PIDNet(m=2, n=3, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=96, head_planes=128, augment=True)
    elif 'pidnet_l' in model_name:
        print("\n >>> Load PIDNet-Large model")
        model = PIDNet(m=3, n=4, num_classes=cfg.DATASET.NUM_CLASSES, planes=64, ppm_planes=112, head_planes=256, augment=True)
    else:
        print(f"\n >>> {model_name} is not supported.")
        raise NotImplementedError
    
    if cfg.MODEL.PRETRAINED != "":
        print(f"Load pretrained weight : {cfg.MODEL.PRETRAINED}")
        pretrained_dict = tf.keras.models.load_model(cfg.MODEL.PRETRAINED)

        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
        logging.info('Attention!!!')
        logging.info(msg)
        logging.info('Over!!!')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict = False)
    else:
        pass
    
    return model

def get_pred_model(name, num_classes):
    model_name = name.lower().replace("-", "_")

    print(f"\n >>> Model name : {name}")

    if 'pidnet_s' in model_name:
        print("\n >>> Load PIDNet-Small model")
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=32, ppm_planes=96, head_planes=128, augment=False)
    elif 'pidnet_m' in model_name:
        print("\n >>> Load PIDNet-Medium model")
        model = PIDNet(m=2, n=3, num_classes=num_classes, planes=64, ppm_planes=96, head_planes=128, augment=False)
    elif 'pidnet_l' in model_name:
        print("\n >>> Load PIDNet-Large model")
        model = PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False)
    else:
        print(f"\n >>> {model_name} is not supported.")
        raise NotImplementedError
    
    return model

if __name__ == "__main__":
    _GPUS = tf.config.experimental.list_physical_devices('GPU')

    model = PIDNet(augment=False, num_classes=19)
    # with tf.stop_gradient():
    input_shape = (1, 1024, 2048, 3)
    x = tf.random.normal(input_shape)
    print(model)

    y = model(x)
    print(y.shape)