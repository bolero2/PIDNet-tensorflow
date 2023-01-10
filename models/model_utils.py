# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, ReLU, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization as BatchNorm

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


BN_MOM = 0.1
ALGC = False

class BasicBlock(tf.Module):
    expansion = 1

    def __init__(self, inplanes, planes, strides=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2D(planes, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn1 = BatchNorm(momentum=BN_MOM)
        self.relu = ReLU()

        self.conv2 = Conv2D(planes, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = BatchNorm(momentum=BN_MOM)

        self.downsample = downsample
        self.strides = strides
        self.no_relu = no_relu

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(tf.Module):
    expansion = 2

    def __init__(self, inplanes, planes, strides=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = BatchNorm(momentum=BN_MOM)

        self.conv2 = Conv2D(planes, kernel_size=3, strides=strides, padding='same', use_bias=False)
        self.bn2 = BatchNorm(momentum=BN_MOM)

        self.conv3 = Conv2D(planes * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = BatchNorm(momentum=BN_MOM)

        self.relu = ReLU()
        self.downsample = downsample
        self.strides = strides
        self.no_relu = no_relu

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class segmenthead(tf.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = BatchNorm(momentum=BN_MOM)
        self.conv1 = Conv2D(interplanes, kernel_size=3, padding='same', use_bias=False)
        self.bn2 = BatchNorm(momentum=BN_MOM)
        self.relu = ReLU()
        self.conv2 = Conv2D(outplanes, kernel_size=1, padding='valid', use_bias=True)
        self.scale_factor = scale_factor

    def __call__(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            width = x.shape[-2] * self.scale_factor
            height = x.shape[-3] * self.scale_factor

            out = tf.image.resize(out, size=[height, width], method='bilinear')

        return out

class DAPPM(tf.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(DAPPM, self).__init__()
        BN_MOM = 0.1
        self.scale1 = Sequential([AveragePooling2D(pool_size=5, strides=2, padding='same'),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                    ])
        self.scale2 = Sequential([AveragePooling2D(pool_size=9, strides=4, padding='same'),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                    ])
        self.scale3 = Sequential([AveragePooling2D(pool_size=17, strides=8, padding='same'),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                    ])
        self.scale4 = Sequential([GlobalAveragePooling2D(),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D( branch_planes, kernel_size=1, use_bias=False),
                                    ])
        self.scale0 = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D( branch_planes, kernel_size=1, use_bias=False),
                                    ])
        self.process1 = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=3, padding='same', use_bias=False),
                                    ])
        self.process2 = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=3, padding='same', use_bias=False),
                                    ])
        self.process3 = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=3, padding='same', use_bias=False),
                                    ])
        self.process4 = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=3, padding='same', use_bias=False),
                                    ])    
        self.compression = Sequential([
                                    BatchNorm(branch_planes * 5, momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(outplanes, kernel_size=1, use_bias=False),
                                    ])
        self.shortcut = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(outplanes, kernel_size=1, use_bias=False),
                                    ])

    def __call__(self, x):
        width = x.shape[-2]
        height = x.shape[-3]        
        x_list = []

        x_list.append(self.scale0(x))
        x_list.append(self.process1((tf.image.resize(self.scale1(x),
                        size=[height, width],
                        method='bilinear')+x_list[0])))
        x_list.append((self.process2((tf.image.resize(self.scale2(x),
                        size=[height, width],
                        method='bilinear')+x_list[1]))))
        x_list.append(self.process3((tf.image.resize(self.scale3(x),
                        size=[height, width],
                        method='bilinear')+x_list[2])))
        x_list.append(self.process4((tf.image.resize(self.scale4(x),
                        size=[height, width],
                        method='bilinear')+x_list[3])))
       
        out = self.compression(tf.concat(x_list, 1)) + self.shortcut(x)
        return out 
    
class PAPPM(tf.Module):
    def __init__(self, inplanes, branch_planes, outplanes):
        super(PAPPM, self).__init__()
        BN_MOM = 0.1
        self.scale1 = Sequential([
                                    AveragePooling2D(pool_size=5, strides=2, padding='same'),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                ])
        self.scale2 = Sequential([
                                    AveragePooling2D(pool_size=9, strides=4, padding='same'),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                ])
        self.scale3 = Sequential([
                                    AveragePooling2D(pool_size=17, strides=8, padding='same'),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                ])
        self.scale4 = Sequential([
                                    GlobalAveragePooling2D(keepdims=True),
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False)
                                ])

        self.scale0 = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(branch_planes, kernel_size=1, use_bias=False),
                                ])
        
        self.scale_process = Sequential([
                                            BatchNorm(momentum=BN_MOM),
                                            ReLU(),
                                            Conv2D(branch_planes * 4, kernel_size=3, padding='same', groups=4, use_bias=False),
                                        ])

      
        self.compression = Sequential([
                                        BatchNorm(momentum=BN_MOM),
                                        ReLU(),
                                        Conv2D(outplanes, kernel_size=1, use_bias=False),
                                    ])
        
        self.shortcut = Sequential([
                                    BatchNorm(momentum=BN_MOM),
                                    ReLU(),
                                    Conv2D(outplanes, kernel_size=1, use_bias=False),
                                ])


    def __call__(self, x):
        width = x.shape[-2]
        height = x.shape[-3]        
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(tf.image.resize(self.scale1(x), size=[height, width],
                        method='bilinear') + x_)
        scale_list.append(tf.image.resize(self.scale2(x), size=[height, width],
                        method='bilinear') + x_)
        scale_list.append(tf.image.resize(self.scale3(x), size=[height, width],
                        method='bilinear') + x_)
        scale_list.append(tf.image.resize(self.scale4(x), size=[height, width],
                        method='bilinear') + x_)
        scale_out = self.scale_process(tf.concat(scale_list, 1))
       
        out = self.compression(tf.concat([x_,scale_out], 1)) + self.shortcut(x)
        return out
    

class PagFM(tf.Module):
    def __init__(self, in_channels, mid_channels, after_relu=False, with_channel=False):
        super(PagFM, self).__init__()
        self.with_channel = with_channel
        self.after_relu = after_relu
        self.f_x = Sequential([
                                Conv2D(mid_channels, kernel_size=1, use_bias=False),
                                BatchNorm()
                            ])
        self.f_y = Sequential([
                                Conv2D(mid_channels, kernel_size=1, use_bias=False),
                                BatchNorm()
                            ])
        if with_channel:
            self.up = Sequential([
                                    Conv2D(in_channels, kernel_size=1, use_bias=False),
                                    BatchNorm()
                                ])
        if after_relu:
            self.relu = ReLU()
        
    def __call__(self, x, y):
        input_size = x.shape

        if self.after_relu:
            y = self.relu(y)
            x = self.relu(x)

        y_q = self.f_y(y)
        y_q = tf.image.resize(y_q, size=[input_size[1], input_size[2]],
                            method='bilinear')
        x_k = self.f_x(x)

        if self.with_channel:
            sim_map = tf.math.sigmoid(self.up(x_k * y_q))
        else:
            sim_map = tf.math.sigmoid(tf.expand_dims(tf.reduce_sum(x_k * y_q, axis=3), 3))
        
        y = tf.image.resize(y, size=[input_size[1], input_size[2]],
                            method='bilinear')

        x = (1 - sim_map) * x + sim_map * y
        
        return x
    
class Light_Bag(tf.Module):
    def __init__(self, in_channels, out_channels):
        super(Light_Bag, self).__init__()
        self.conv_p = Sequential([
                                    Conv2D(out_channels, kernel_size=1, use_bias=False),
                                    BatchNorm()
                                ])
        self.conv_i = Sequential([
                                    Conv2D(out_channels, kernel_size=1, use_bias=False),
                                    BatchNorm()
                                ])
        
    def __call__(self, p, i, d):
        edge_att = tf.math.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add
    

class DDFMv2(tf.Module):
    def __init__(self, in_channels, out_channels):
        super(DDFMv2, self).__init__()
        self.conv_p = Sequential([
                                    BatchNorm(),
                                    ReLU(),
                                    Conv2D(out_channels, kernel_size=1, use_bias=False),
                                    BatchNorm()
                                ])
        self.conv_i = Sequential([
                                    BatchNorm(),
                                    ReLU(),
                                    Conv2D(out_channels, kernel_size=1, use_bias=False),
                                    BatchNorm()
                                ])
        
    def __call__(self, p, i, d):
        edge_att = tf.math.sigmoid(d)
        
        p_add = self.conv_p((1-edge_att)*i + p)
        i_add = self.conv_i(i + edge_att*p)
        
        return p_add + i_add

class Bag(tf.Module):
    def __init__(self, in_channels, out_channels):
        super(Bag, self).__init__()

        self.conv = Sequential([
                                BatchNorm(),
                                ReLU(),
                                Conv2D(in_channels, out_channels, 
                                          kernel_size=3, padding='same', use_bias=False)                  
                                ])

        
    def __call__(self, p, i, d):
        edge_att = tf.math.sigmoid(d)
        return self.conv(edge_att*p + (1-edge_att)*i)
    


if __name__ == '__main__':
    x = tf.random.normal((4, 64, 32, 64))
    y = tf.random.normal((4, 64, 32, 64))
    z = tf.random.normal((4, 64, 32, 64))
    net = PagFM(64, 16, with_channel=True)
    
    out = net(x, y)
    print("\n out :", out)