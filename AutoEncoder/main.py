from DeepAE import *
import tensorflow_datasets as tfds


def map_image(image, label):
    """Normalizes and flattens the image. Returns image as input and label."""
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.reshape(image, shape=(784,))

    return image, image


def get_data():
    # Load the train and test sets from TFDS

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 1024

    train_dataset = tfds.load('mnist', as_supervised=True, split="train")
    train_dataset = train_dataset.map(map_image)
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    test_dataset = tfds.load('mnist', as_supervised=True, split="test")
    test_dataset = test_dataset.map(map_image)
    test_dataset = test_dataset.batch(BATCH_SIZE).repeat()
    return train_dataset, test_dataset, BATCH_SIZE


if __name__ == '__main__':
    # train, test, BATCH_SIZE = get_data()

    # train_steps = 60000 // BATCH_SIZE
    train = None
    model = DeepAE(784, [64, 32, 16, 4])
    model.build_model()
    model.fit(train, tf.keras.optimizers.Adam(), 'binary_crossentropy', epochs=50)
