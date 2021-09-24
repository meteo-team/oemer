import tensorflow as tf
import tensorflow.keras.layers as L


def naive_conv(win_size=256):
    inp = L.Input(shape=(win_size, win_size, 3))
    tensor = tf.cast(inp, tf.float32)
    conv1 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(tensor)
    conv1 = L.LayerNormalization(dtype=tf.float32)(conv1)
    conv1 = L.Dropout(0.3)(conv1)

    conv2 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv1)
    conv2 = L.LayerNormalization(dtype=tf.float32)(conv2)
    conv2 = L.Dropout(0.3)(conv2)

    conv3 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv2)
    conv3 = L.LayerNormalization(dtype=tf.float32)(conv3)
    conv3 = L.Dropout(0.3)(conv3)

    conv4 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv3)
    conv4 = L.LayerNormalization(dtype=tf.float32)(conv4)
    conv4 = L.Dropout(0.3)(conv4)

    conv5 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv4)
    conv5 = L.LayerNormalization(dtype=tf.float32)(conv5)
    conv5 = L.Dropout(0.3)(conv5)

    conv6 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv5)
    conv6 = L.LayerNormalization(dtype=tf.float32)(conv6)
    conv6 = L.Dropout(0.3)(conv6)

    conv7 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv5)
    conv7 = L.LayerNormalization(dtype=tf.float32)(conv7)
    conv7 = L.Dropout(0.3)(conv7)

    conv8 = L.Conv2D(128, (3, 3), activation='swish', padding='same', dtype=tf.float32)(conv7)
    conv8 = L.LayerNormalization(dtype=tf.float32)(conv8)
    conv8 = L.Dropout(0.3)(conv8)

    out = L.Conv2D(3, (3, 3), activation='sigmoid', padding='same', dtype=tf.float32)(conv8)
    return tf.keras.Model(inputs=inp, outputs=out)
