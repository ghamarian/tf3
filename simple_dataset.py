import tensorflow as tf
import csv

with open('input.csv', 'r') as csvfile:
    csv_columns = csv.reader(csvfile, delimiter=',')
    feature_names = next(csv_columns)

epoch_number = 3
batch_size = 32

def decode(line, label_name):
    parsed_line = tf.decode_csv(line, record_defaults=[[0], ['house'], [0]])
    features = dict(zip(feature_names, parsed_line))
    labels = features.pop(label_name)

    return features, labels


dataset = tf.data.TextLineDataset("input.csv").skip(1)
dataset = dataset.map(lambda x: decode(x, 'price'))
d = dataset.repeat(epoch_number).batch(batch_size).shuffle(buffer_size=1000).make_one_shot_iterator().get_next()

with tf.Session() as sess:
    print(sess.run(d))
