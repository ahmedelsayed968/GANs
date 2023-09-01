import time

from utils import *
import tensorflow_hub as hub
import tensorflow as tf


def load_model(url: str):
    hub_module = hub.load(url)
    return hub_module


def get_image_stylized(content_image, style_image, model):
    if model:
        return model(tf.image.convert_image_dtype(content_image, dtype=tf.float32),
                     tf.image.convert_image_dtype(style_image, dtype=tf.float32))
    else:
        print('Failded')


if __name__ == '__main__':
    IMAGE_DIR = './images'
    content_path = f'{IMAGE_DIR}/swan-2107052_1280.jpg'
    style_path = f'{IMAGE_DIR}/Vassily_Kandinsky,_1913_-_Composition_7.jpg'
    content_image, style_image = load_images(content_path, style_path)
    URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    model = load_model(URL)
    outputs = get_image_stylized(content_image,
                                 style_image,
                                 model)
    stylized_image_ts = outputs[0]
    stylized_image = tensor_to_image(stylized_image_ts)
    finished_time = time.time()
    stylized_image.save(f'E:\\GANs\\Style-Transfer\\output\\image-{finished_time}.jpg')
