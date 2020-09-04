import tensorflow as tf

inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
# outputs = embedding(inputs, 6, 2, zero_pad=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print (sess.run(inputs))