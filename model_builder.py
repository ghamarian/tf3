import tensorflow as tf
from importlib import import_module
from inspect import Parameter, signature, getfullargspec

import pprint


class ModelBuilder:

    def __init__(self, features):
        self.features = features
        self.cls = tf.estimator.Estimator
        self.subclasses = self._all_subclasses(self.cls)
        self.positional = self.positional_arguments()
        self.none_args = self.none_arguments()
        self.singatuer = self.signature_list()

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

    def _all_subclasses(self, cls):
        return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                       for g in self._all_subclasses(s)]

    def subclasses_name_list(self):
        return [self.method_name(cls) for cls in self.subclasses]

    def method_name(self, cls):
        return str(cls).rsplit('.', 1)[1][:-2]

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

    def _arguments_with(self, predicate):
        all_args = {}
        for cls in self.subclasses:
            args = []
            for x, p in signature(cls).parameters.items():
                if predicate(p.default) and p.kind != Parameter.VAR_POSITIONAL:
                    args.append(x)
            all_args.update({self.method_name(cls): args})

        return all_args

    def positional_arguments(self):
        return self._arguments_with(lambda p: p == Parameter.empty)

    def none_arguments(self):
        return self._arguments_with(lambda p: p == Parameter.empty or p == None)

    def signature_list(self):
        args = [getfullargspec(estimator).args for estimator in self.subclasses]
        for arg in args:
            arg.remove('self')

        return  list(zip(self.subclasses_name_list(), args))


# print("\n".join([str(key) for key in inheritors(tf.estimator.Estimator)]))


# print(ModelBuilder([]).subclasses_name_list(tf.estimator.Estimator))
# mb = ModelBuilder([])
# plist = mb.parameters_list(tf.estimator.Estimator)
# print(pprint.pformat(plist))

# print(pprint.pformat(mb.signature_list(), width=200))

# print("\n".join([str(l) for l in mb.parameters_list(tf.estimator.Estimator)]))


# print(len(ModelBuilder([]).estimator_class_list(tf.estimator.Estimator)))
#
# print(len(ModelBuilder([]).all_subclasses(tf.estimator.Estimator)))
# grab("tensorflow.python.estimator.canned.LinearRegressor", )
