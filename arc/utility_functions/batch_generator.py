"""
batch_generator.py
"""

import os, random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical as tocat_fn

Image.LOAD_TRUNCATED_IMAGES = True


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_list, label_list, batch_size, image_size=(150, 150), aug_flag=False):
        self.data_list = data_list
        self.label_list = label_list
        self.batch_size = batch_size
        self.image_size = image_size
        self.aug_flag = aug_flag

        self.total_images = len(self.data_list)
        self.indices = np.arange(self.total_images)
        self.num_batches = int(np.ceil(self.total_images/self.batch_size))
        #self.on_epoch_end()

    def __len__(self):
        """ iterations per epoch """
        return self.num_batches

    def on_epoch_end(self):
        random.shuffle(self.indices)

    def __getitem__(self, index):
        """ return batch of (data, label) pairs """
        batch_x, batch_y = [], []
        batch_indices = self.indices[index*self.batch_size:min((index+1)*self.batch_size, self.total_images)]

        for loop in batch_indices:
            loaded_image = img_to_array((load_img(os.path.join(
                self.data_list[loop]))).resize(self.image_size, Image.ANTIALIAS))
            loaded_label = tocat_fn(self.label_list[loop], 100)

            if self.aug_flag:
                loaded_image = self._random_rotate(loaded_image)

            batch_x.append(loaded_image)
            batch_y.append(loaded_label)

        return (np.asarray(batch_x, dtype=np.float32),
                np.asarray(batch_y, dtype=np.uint8))

    def _random_augment(self, image):
        if np.random.uniform(-1, 1) > 0:
            return self._random_rotate(image)
        else:
            return self._random_brightness_distort(image)

    @staticmethod
    def _random_rotate(image):
        angle_multiplier = np.random.randint(3)
        return np.rot90(image, angle_multiplier)

    @staticmethod
    def _random_brightness_distort(image):
        noise_shift = np.random.normal(0., .05, image.shape)
        noise_scale = np.random.normal(1., .01, image.shape)
        return (image + noise_shift) * noise_scale
