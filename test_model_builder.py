import pytest
import tensorflow as tf

from model_builder import ModelBuilder
import pprint
import numpy as np

@pytest.fixture
def model_builder():
    return ModelBuilder([])


def output(a):
    print(pprint.pformat(a))


# SUB_CLASSES = [tf.python.estimator.canned.baseline.BaselineClassifier,
#  tf.python.estimator.canned.baseline.BaselineRegressor,
#  tf.python.estimator.canned.boosted_trees.BoostedTreesClassifier,
#  tf.python.estimator.canned.boosted_trees.BoostedTreesRegressor,
#  tf.python.estimator.canned.dnn.DNNClassifier,
#  tf.python.estimator.canned.dnn.DNNRegressor,
#  tf.python.estimator.canned.linear.LinearClassifier,
#  tf.python.estimator.canned.linear.LinearRegressor,
#  tf.python.estimator.canned.dnn_linear_combined.DNNLinearCombinedClassifier,
#  tf.python.estimator.canned.dnn_linear_combined.DNNLinearCombinedRegressor]

NAME_LIST = ['BaselineClassifier', 'BaselineRegressor', 'BoostedTreesClassifier', 'BoostedTreesRegressor',
             'DNNClassifier', 'DNNRegressor', 'LinearClassifier', 'LinearRegressor', 'DNNLinearCombinedClassifier',
             'DNNLinearCombinedRegressor']

POSITIONAL = ({'BaselineClassifier': [], 'BaselineRegressor': [],
               'BoostedTreesClassifier': ['feature_columns', 'n_batches_per_layer'],
               'BoostedTreesRegressor': ['feature_columns', 'n_batches_per_layer'],
               'DNNClassifier': ['hidden_units', 'feature_columns'], 'DNNLinearCombinedClassifier': [],
               'DNNLinearCombinedRegressor': [], 'DNNRegressor': ['hidden_units', 'feature_columns'],
               'LinearClassifier': ['feature_columns'], 'LinearRegressor': ['feature_columns']})

NONE_ARGS = {'BaselineClassifier': ['model_dir',
                                    'weight_column',
                                    'label_vocabulary',
                                    'config'],
             'BaselineRegressor': ['model_dir', 'weight_column', 'config'],
             'BoostedTreesClassifier': ['feature_columns',
                                        'n_batches_per_layer',
                                        'model_dir',
                                        'weight_column',
                                        'label_vocabulary',
                                        'config'],
             'BoostedTreesRegressor': ['feature_columns',
                                       'n_batches_per_layer',
                                       'model_dir',
                                       'weight_column',
                                       'config'],
             'DNNClassifier': ['hidden_units',
                               'feature_columns',
                               'model_dir',
                               'weight_column',
                               'label_vocabulary',
                               'dropout',
                               'input_layer_partitioner',
                               'config',
                               'warm_start_from'],
             'DNNLinearCombinedClassifier': ['model_dir',
                                             'linear_feature_columns',
                                             'dnn_feature_columns',
                                             'dnn_hidden_units',
                                             'dnn_dropout',
                                             'weight_column',
                                             'label_vocabulary',
                                             'input_layer_partitioner',
                                             'config',
                                             'warm_start_from'],
             'DNNLinearCombinedRegressor': ['model_dir',
                                            'linear_feature_columns',
                                            'dnn_feature_columns',
                                            'dnn_hidden_units',
                                            'dnn_dropout',
                                            'weight_column',
                                            'input_layer_partitioner',
                                            'config',
                                            'warm_start_from'],
             'DNNRegressor': ['hidden_units',
                              'feature_columns',
                              'model_dir',
                              'weight_column',
                              'dropout',
                              'input_layer_partitioner',
                              'config',
                              'warm_start_from'],
             'LinearClassifier': ['feature_columns',
                                  'model_dir',
                                  'weight_column',
                                  'label_vocabulary',
                                  'config',
                                  'partitioner',
                                  'warm_start_from'],
             'LinearRegressor': ['feature_columns',
                                 'model_dir',
                                 'weight_column',
                                 'config',
                                 'partitioner',
                                 'warm_start_from']}


def test_all_subclasses(model_builder):
    output(model_builder.subclasses)
    output(model_builder.subclasses_name_list())
    np.testing.assert_array_equal(model_builder.subclasses_name_list(), NAME_LIST)
    positional = model_builder.positional_arguments(),
    np.testing.assert_array_equal(positional, POSITIONAL)
    none = model_builder.none_arguments()
    np.testing.assert_array_equal(none, NONE_ARGS)
    all_args = model_builder.signature_dict()
    # output(all_args)
    print(model_builder.none_args_of('DNNRegressor'))
    print(model_builder.all_args_of('DNNRegressor'))
    print(model_builder.positional_args_of('DNNRegressor'))
    # name_class_dict = model_builder.name_class_dict
    # output(name_class_dict['DNNRegressor'])
    output(model_builder.actual_class_of('DNNRegressor'))


@pytest.fixture
def none():
   return ['hidden_units', 'feature_columns', 'model_dir', 'weight_column', 'dropout', 'input_layer_partitioner',
           'config', 'warm_start_from']

@pytest.fixture
def positional():
    return ['hidden_units', 'feature_columns']

@pytest.fixture
def all():
    return ['hidden_units', 'feature_columns', 'model_dir', 'label_dimension', 'weight_column', 'optimizer',
            'activation_fn', 'dropout', 'input_layer_partitioner', 'config', 'warm_start_from', 'loss_reduction']


@pytest.fixture
def args(all):
    from itertools import count
    return dict(zip(all, count()))


def test_check_args(model_builder, positional, none, args):
    assert model_builder.check_args('DNNRegressor', positional, args)

def test_less_positional(model_builder, positional, none, args):
    positional.pop(0)
    assert not model_builder.check_args('DNNRegressor',  positional, args)

def test_too_many(model_builder, positional, none, args):
    args.update({'amir': 12})
    assert not model_builder.check_args('DNNRegressor',  positional, args)
