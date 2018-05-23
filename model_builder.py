import tensorflow as tf
from importlib import import_module

class ModelBuilder:

    def __init__(self, features):
       self.features = features

    def estimator_class_list(self, klass):
      subclasses = set()
      work = [klass]
      while work:
         parent = work.pop()
         for child in parent.__subclasses__():
            if child not in subclasses:
               subclasses.add(child)
               work.append(child)
      return subclasses


    def create(self, estimator_name, *args, **kwargs):

        try:
            module_name, class_name = estimator_name.rsplit('.', 1)
            estimator_module = import_module(module_name)
            estimator_class = getattr(estimator_module, class_name)
            instance = estimator_class(*args, **kwargs)

        except (AttributeError, ModuleNotFoundError):
            raise ImportError('{} is not an estimator!'.format(estimator_name))
        else:
            if not issubclass(estimator_class, tf.estimator.Estimator):
                raise ImportError('{} is not an estimator!'.format(estimator_name))

        return instance


# print("\n".join([str(key) for key in inheritors(tf.estimator.Estimator)]))

# grab("tensorflow.python.estimator.canned.LinearRegressor", )