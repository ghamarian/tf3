import pandas as pd
import numpy as np

import tensorflow as tf


def create_data():
    global i
    a = pd.DataFrame()
    b = pd.DataFrame()
    a['col'] = [f'A{i}' for i in range(100)]
    b['col'] = [f'B{i}' for i in range(100)]
    a.to_csv('data/data-a.csv', index=False)
    b.to_csv('data/data-b.csv', index=False)


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.DEBUG)

# def make_csv_dataset(
#         file_pattern,
#         batch_size,
#         column_names=None,
#         column_defaults=None,
#         label_name=None,
#         field_delim=",",
#         use_quote_delim=True,
#         na_value="",
#         header=True,
#         comment=None,
#         num_epochs=None,
#         shuffle=True,
#         shuffle_buffer_size=10000,
#         shuffle_seed=None,
#         prefetch_buffer_size=1,
#         num_parallel_reads=1,
#         num_parallel_parser_calls=2,
#         sloppy=False,
#         default_float_type=dtypes.float32,
#         num_rows_for_inference=100,
# ):

dataset = tf.contrib.data.make_csv_dataset("data/data-*.csv", 4, num_epochs=1, label_name="col", num_parallel_reads=4, shuffle=False)

for batch in dataset:
    print(batch)
