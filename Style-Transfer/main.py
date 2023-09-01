from neural_style_transfer_custom import *
import time
from utils import *

if __name__ == '__main__':
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layers = ['block5_conv2']

    IMAGE_DIR = './images'
    content_path = f'{IMAGE_DIR}/swan-2107052_1280.jpg'
    style_path = f'{IMAGE_DIR}/Vassily_Kandinsky,_1913_-_Composition_7.jpg'
    content_image, style_image = load_images(content_path, style_path)

    output_layers = style_layers + content_layers
    num_style_layers: int = len(style_layers)
    num_content_layers: int = len(content_layers)
    weights = 'imagenet'
    model = VGG19(weights,
                  num_style_layers,
                  num_content_layers,
                  output_layers)
    model.build_model()
    optimizer = tf.optimizers.Adam()
    generated_image, final_image = model.fit_style_transfer(style_image,
                                                            content_image,
                                                            optimizer,
                                                            epochs=20,
                                                            steps_per_epoch=100)
    print('finished')
    finished_time = time.time()
    styled_image = tensor_to_image(final_image)
    styled_image.save(f'E:\\GANs\\Style-Transfer\\output\\image-{finished_time}.jpg')
