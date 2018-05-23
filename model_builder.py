import tensorflow as tf
from importlib import import_module

class ModelBuilder:

    def __init__(self, features):
       self.features = features

    def inheritors(self, klass):
      subclasses = set()
      work = [klass]
      while work:
         parent = work.pop()
         for child in parent.__subclasses__():
            if child not in subclasses:
               subclasses.add(child)
               work.append(child)
      return subclasses


    def grab(self, estimator_name, *args, **kwargs):

        try:
            if '.' in estimator_name:
                module_name, class_name = estimator_name.rsplit('.', 1)
            else:
                module_name = estimator_name
                class_name = estimator_name.capitalize()

            estimator_module = import_module(module_name)

            estimator_class = getattr(estimator_module, class_name)

            instance = estimator_class(*args, **kwargs)

        except (AttributeError, ModuleNotFoundError):
            raise ImportError('{} is not part of our animal collection!'.format(estimator_name))
        else:
            if not issubclass(estimator_class, tf.estimator.Estimator):
                raise ImportError("We currently don't have {}, but you are welcome to send in the request for it!".format(estimator_class))

        return instance


# print("\n".join([str(key) for key in inheritors(tf.estimator.Estimator)]))

# grab("tensorflow.python.estimator.canned.LinearRegressor", )