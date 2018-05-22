import tensorflow as tf

tf.enable_eager_execution()


A = tf.constant([[1, 2], [2, 1], [-2, -1]])

print(A)

with tf.Session() as sess:
    print(sess.run([A]))

# print(tf.tile(A, [2, 3]))

