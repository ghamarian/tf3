from utils import feature_util


def test_already_order_reorder_request():
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    categories = ['numerical', 'numerical', 'numerical', 'numerical', 'categorical']
    defaults = ['5.8', '3', '4.35', '1.3', 'Iris-setosa']
    list_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    new_categories , new_defaults = feature_util.reorder_request(features, categories, defaults, list_features)
    assert categories == new_categories
    assert defaults == new_defaults


def test_disorder_reorder_request():
    features = ['sepal_length', 'petal_length', 'petal_width', 'class', 'sepal_width']
    categories = ['numerical', 'numerical', 'numerical', 'categorical', 'numerical']
    defaults = ['5.8', '4.35', '1.3', 'Iris-setosa','3']
    list_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    new_categories , new_defaults = feature_util.reorder_request(features, categories, defaults, list_features)
    assert ['numerical', 'numerical', 'numerical', 'numerical', 'categorical'] == new_categories
    assert ['5.8', '3', '4.35', '1.3', 'Iris-setosa'] == new_defaults


def test_remove_target():
    features = ['sepal_length', 'petal_length', 'petal_width', 'class', 'sepal_width']
    target = 0
    new_features = feature_util.remove_target(features, target)
    assert features[0] not in new_features

def test_remove_target():
    features = ['sepal_length', 'petal_length', 'petal_width', 'class', 'sepal_width']
    target = 9
    new_features = feature_util.remove_target(features, target)
    assert new_features == features