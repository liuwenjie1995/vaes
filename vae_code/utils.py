import numpy as np
import tensorflow as tf


def merge(images, size):
    """
    融合
    :param images:图片的数量
    :param size:图片的尺寸
    :return:一张融合过后的图像
    """
    h, w = images.shape[-1], images.shape[2]
    img = np.zeros([h * size[0], w * size[1]])

    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx % size[1]
        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w)] = image

    return img
