import tensorflow as tf


def content_loss(generated_features, target_features):
    return 0.5 * tf.reduce_sum(tf.square(generated_features - target_features))


def style_loss(generated_features, target_features):
    return tf.reduce_mean(tf.square(generated_features - target_features))


def gram_matrix(input_tensor):
    gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    height, width = input_shape[1], input_shape[2]

    # get number of cells in the input image to normalize the gram
    num_locations = tf.cast(height * width, tf.float32)
    return gram / num_locations


def preprocess_image(image):
    return tf.keras.applications.vgg19.preprocess_input(image)


def clip_image_values(image, min_value=0.0, max_value=255.0):
    """clips the image pixel values by the given min and max"""
    return tf.clip_by_value(image, clip_value_min=min_value, clip_value_max=max_value)


class VGG19:
    def __init__(self,
                 weights: str,
                 num_style_layers: int,
                 num_content_layers: int,
                 output_layers_names: list[str]):
        self.Model = None
        self.weights = weights
        self.output_layers = output_layers_names
        self.input = None
        self.NUM_STYLE_LAYERS = num_style_layers
        self.NUM_CONTENT_LAYERS = num_content_layers
        # self.image = image

    def build_model(self):
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                weights=self.weights)
        vgg.trainable = False
        outputs = [vgg.get_layer(layer).output for layer in self.output_layers]
        for layer in outputs:
            print(layer)

        self.Model = tf.keras.Model(inputs=vgg.input, outputs=outputs)

    def get_style_image_features(self, image):
        if self.Model is None:
            return
        preprocessed_image = preprocess_image(image)
        outputs_model = self.Model(preprocessed_image)

        # filter only the style layers
        style_layers = outputs_model[:self.NUM_STYLE_LAYERS]
        # get the gram of each output tensor from the style layers
        gram_style_features = [gram_matrix(layer) for layer in style_layers]
        return gram_style_features

    def get_content_image_features(self, image):
        if self.Model is None:
            return
        preprocessed_image = preprocess_image(image)
        outputs = self.Model(preprocessed_image)
        content_tensors = outputs[self.NUM_STYLE_LAYERS:]
        return content_tensors

    def get_total_loss(self,
                       content_targets,
                       content_generated,
                       style_targets,
                       style_generated,
                       content_weight,
                       style_weight):

        style_loss_total = tf.add_n([style_loss(generated, target)
                                     for target, generated in zip(style_targets, style_generated)])

        content_loss_total = tf.add_n([content_loss(generated, target)
                                       for target, generated in zip(content_targets, content_generated)])

        total_loss = (style_loss_total * style_weight) / self.NUM_STYLE_LAYERS \
                     + (content_loss_total * content_weight) / self.NUM_CONTENT_LAYERS
        return total_loss

    def calculate_gradients(self,
                            image,
                            style_targets,
                            content_targets,
                            style_weight,
                            content_weight):
        with tf.GradientTape() as tape:
            style_features = self.get_style_image_features(image)
            content_features = self.get_content_image_features(image)
            loss = self.get_total_loss(content_targets,
                                       content_features,
                                       style_targets,
                                       style_features,
                                       content_weight,
                                       style_weight)

        gradients = tape.gradient(loss, image)
        return gradients

    def update_image(self,
                     image,
                     style_targets,
                     content_targets,
                     style_weight,
                     content_weight,
                     optimizer: tf.keras.optimizers):
        gradients = self.calculate_gradients(image,
                                             style_targets,
                                             content_targets,
                                             style_weight,
                                             content_weight)
        gradients_and_vars = zip(gradients, [image])  # Create a list of gradient-variable pairs
        optimizer.apply_gradients(gradients_and_vars)
        image.assign(clip_image_values(image))

    def fit_style_transfer(self,
                           style_image,
                           content_image,
                           optimizer,
                           epochs,
                           steps_per_epoch,
                           style_weight=1e-2,
                           content_weight=1e-4
                           ):
        if self.Model is None:
            return

        generated_images = []
        step = 0

        # get our targets
        content_targets = self.get_content_image_features(content_image)
        style_targets = self.get_style_image_features(style_image)

        # initialize the generated image and make it variable to be updated
        generated_image = tf.cast(content_image, tf.float32)
        generated_image = tf.Variable(generated_image)

        generated_images.append(generated_image)
        for epoch in range(epochs):
            for step_ in range(steps_per_epoch):
                self.update_image(generated_image,
                                  style_targets,
                                  content_targets,
                                  style_weight,
                                  content_weight,
                                  optimizer)
                if epoch % 10 == 0:
                    generated_images.append(generated_image)

        generated_image = tf.cast(generated_image, tf.uint8)
        return generated_images, generated_image


if __name__ == '__main__':
    pass
