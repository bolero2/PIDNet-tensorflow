from tensorflow.keras.utils import Sequence
import math
import numpy as np
import os
from PIL import Image
import cv2
import random


IMAGE_EXT = 'jpg'
LABEL_EXT = 'png'

y_k_size = 6
x_k_size = 6

class CustomDataset(Sequence):
    def __init__(self, config,
                 datapack:str,
                 batch_size:int,
                 shuffle:bool=False,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=255, 
                 base_size=640, 
                 crop_size=(480, 640),
                 reshape_size=(640, 480),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4,
                 mode=''
                ):

        assert mode in ['train', 'valid', 'test']
        self.mode = mode

        self.config = config
        self.datapack = datapack

        self.image_prefix = os.path.join(config.DATASET.ROOT, self.mode, config.DATASET.IMAGES)
        self.label_prefix = os.path.join(config.DATASET.ROOT, self.mode, config.DATASET.ANNOTATIONS)


        self.multi_scale = multi_scale
        self.flip = flip
        self.ignore_label = ignore_label

        self.base_size = base_size
        self.crop_size = crop_size
        self.reshape_size = reshape_size
        self.scale_factor = scale_factor

        self.mean = mean
        self.std = std
        self.bd_dilate_size = bd_dilate_size

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.datapack) / self.batch_size)

    def __getitem__(self, index):
        """
        Pytorch와 다르게, index 항목에 있는 1개만 return 하는 것이 아닌
        index 번 째의 batch_size 만큼을 return해줘야 함.
        """
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        imagepaths = [os.path.join(self.image_prefix, f"{self.datapack[x]}.{IMAGE_EXT}") for x in indices]
        labelpaths = [os.path.join(self.label_prefix, f"{self.datapack[x]}.{LABEL_EXT}") for x in indices]

        for imgfile in imagepaths:
            assert os.path.isfile(imgfile)
            x = Image.open(imgfile).convert("RGB")
            x = x.resize(self.reshape_size)
            x = np.array(x)
            size = x.shape

        for labelfile in labelpaths:
            assert os.path.isfile(labelfile)
            y = Image.open(labelfile)
            y = y.resize(self.reshape_size)
            y = np.array(y)

        image, label, edge = self.gen_sample(x, y, 
                                self.multi_scale, self.flip, edge_pad=False,
                                edge_size=self.bd_dilate_size)
        _image = image.copy()
        _label = label.copy()
        _edge = edge.copy()

        return _image, _label, _edge, np.array(size), [self.datapack[x] for x in indices]

    def on_epoch_end(self):
        self.indices = np.arange(len(self.datapack))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def gen_sample(self, image, label,
                   multi_scale=True, is_flip=True, edge_pad=True, edge_size=4, city=False):
        
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-1 * y_k_size, x_k_size:-1 * x_k_size]
            edge = np.pad(edge, ((y_k_size, y_k_size),(x_k_size, x_k_size)), mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0
        
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label, edge = self.multi_scale_aug(image, label, edge,
                                                rand_scale=rand_scale)

        image = self.input_transform(image)
        label = self.label_transform(label)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge

    def input_transform(self, image):
        image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def label_transform(self, label):
        return np.array(label).astype(np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0,
                                           pad_w, cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size,
                               (0.0, 0.0, 0.0))
        label = self.pad_image(label, h, w, self.crop_size,
                               (self.ignore_label,))
        edge = self.pad_image(edge, h, w, self.crop_size,
                               (0.0,))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        edge = edge[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label, edge

    def multi_scale_aug(self, image, label=None, edge=None,
                        rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge
