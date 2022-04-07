import tensorflow as tf
from qkeras import *
import numpy


def train(first, second, third, fourth, fifth, batch_size=256, epochs=30):
    tf.keras.backend.clear_session()
    batch_size = batch_size
    epochs = epochs
    lr = 0.01
    decay = 0.0001

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    n_train = train_images.shape[0]
    n_test = test_images.shape[0]
    num_classes = 10

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    train_images = train_images/255.0
    test_images = test_images/255.0

    train_images = train_images.reshape((n_train, 28, 28, 1))
    test_images = test_images.reshape((n_test, 28, 28, 1))

    train_images = tf.keras.layers.ZeroPadding2D(padding=2)(train_images)
    test_images = tf.keras.layers.ZeroPadding2D(padding=2)(test_images)

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(n_train).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)

    model = tf.keras.models.Sequential()
    model.add(QConv2D(filters=6, kernel_size=(3, 3),
                      kernel_quantizer=quantized_bits(8, 0, 0),
                      bias_quantizer=quantized_bits(32, 0, 0),
                      input_shape=(32, 32, 1)))
    model.add(QActivation(first))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(QConv2D(filters=16, kernel_size=(3, 3),
                      kernel_quantizer=quantized_bits(8, 0, 0),
                      bias_quantizer=quantized_bits(32, 0, 0),
                      activation='relu'))
    model.add(QActivation(second))
    model.add(tf.keras.layers.AveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(QDense(units=120,
                     kernel_quantizer=quantized_bits(8, 0, 0),
                     bias_quantizer=quantized_bits(32, 0, 0)))
    model.add(QActivation(third))
    model.add(QDense(units=84,
                     kernel_quantizer=quantized_bits(8, 0, 0),
                     bias_quantizer=quantized_bits(32, 0, 0)))
    model.add(QActivation(fourth))
    model.add(QDense(units=10, activation='softmax',
                     kernel_quantizer=quantized_bits(8, 0, 0),
                     bias_quantizer=quantized_bits(32, 0, 0)))

    loss_object = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, decay=decay)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    model.compile(optimizer, loss_object, train_accuracy)

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)


    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)

    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

    # test_loss, test_acc = model.evaluate(test_images, test_labels)
    return train_loss.result().numpy(), \
           train_accuracy.result().numpy(), \
           test_loss.result().numpy(), \
           test_accuracy.result().numpy()
