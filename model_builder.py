import tensorflow as tf
from importlib import import_module
from inspect import Parameter, signature

import pprint


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
               print(child)
               subclasses.add(child)
               work.append(child)
      return subclasses

    def all_subclasses(self, cls):
        return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                       for g in self.all_subclasses(s)]

    def subclasses_name_list(self, cls):
        return [str(cls).rsplit('.', 1)[1][:-2] for cls in self.all_subclasses(cls)]

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

    def parameters_list(self, cls):
       a = self.all_subclasses(cls)
       all_args = { }
       for foo in a:
           args = []
           for x, p in signature(foo).parameters.items():
               if (p.default == Parameter.empty or p.default == None) and p.kind != Parameter.VAR_POSITIONAL:
                   args.append(x)
           all_args.update({foo: args})

       return all_args

       # return [ k for estimator in a for k in inspect.signature(estimator).parameters[] if Parameter.kind != .kind]
       # return [ inspect.getfullargspec(estimator) for estimator in a ]


# print("\n".join([str(key) for key in inheritors(tf.estimator.Estimator)]))


# print(ModelBuilder([]).subclasses_name_list(tf.estimator.Estimator))
mb = ModelBuilder([])
plist = mb.parameters_list(tf.estimator.Estimator)
print(pprint.pformat(plist))


# print("\n".join([str(l) for l in mb.parameters_list(tf.estimator.Estimator)]))



# print(len(ModelBuilder([]).estimator_class_list(tf.estimator.Estimator)))
#
# print(len(ModelBuilder([]).all_subclasses(tf.estimator.Estimator)))
# grab("tensorflow.python.estimator.canned.LinearRegressor", )