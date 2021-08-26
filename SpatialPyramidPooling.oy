import tensorflow as tf


def SpatialPyramidPooling(previous_conv, out_pool_size_list):
    b, w, h, c = previous_conv.shape
    for index, pool_size in enumerate(out_pool_size_list):
        w_wid = tf.cast(tf.math.ceil(w / pool_size), tf.int64)
        h_wid = tf.cast(tf.math.ceil(h / pool_size), tf.int64)
        max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(w_wid, h_wid), strides=(w_wid, h_wid), padding='same')
        result = tf.reshape(max_pooling(previous_conv), (b, -1))
        spp = result if index == 0 else tf.concat([spp, result], axis=-1)
    return spp


if __name__ == '__main__':
    inputs = tf.random.normal(shape=(8, 19, 14, 256))
    result = SpatialPyramidPooling(inputs, out_pool_size_list=[4, 2, 1])
    print(result.shape)
