import tensorflow as tf


dataset = tf.data.Dataset.range(100)

dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
# dataset = dataset.padded_batch(4, padded_shapes=[None])

iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()


with tf.Session() as sess:
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))
    print(sess.run(next_element))


