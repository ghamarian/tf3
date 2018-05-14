import tensorflow as tf
import itertools

# a = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
a = tf.constant([1, 2, 3, 4, 1, 2, 3, 4])

b = tf.split(a, 4)

nds = tf.data.Dataset.from_generator(itertools.count, tf.int32)

k  = nds.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(6), cycle_length=2, block_length=4).make_one_shot_iterator()
l = nds.batch(10).make_one_shot_iterator()



with tf.Session() as sess:
    # print(sess.run(b))
    for i in range(100):
        # print(sess.run(k.get_next()))
        print(sess.run(l.get_next()))

