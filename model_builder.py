import tensorflow as tf
from importlib import import_module
from inspect import Parameter, signature, getfullargspec
from collections import OrderedDict

class ModelBuilder:

    def __init__(self, features):
        self.features = features
        self.cls = tf.estimator.Estimator
        self.subclasses = self._all_subclasses(self.cls)
        self.subclasses_names = self.subclasses_name_list()
        self.positional = self.positional_arguments()
        self.none_args = self.none_arguments()
        self.all_args = self.signature_dict()
        self.name_class_dict = self.create_name_class_dict()

    def actual_class_of(self, name):
        return self.name_class_dict[name]

    def create_name_class_dict(self):
        return OrderedDict(zip(self.subclasses_names, self.subclasses))

    def positional_args_of(self, name):
        return self.positional[name]

    def none_args_of(self, name):
        return self.none_args[name]

    def all_args_of(self, name):
        return self.all_args[name]

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

            assert self.check_args(estimator_name, args, kwargs)

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

    def signature_dict(self):
        args = [getfullargspec(estimator).args for estimator in self.subclasses]
        for arg in args:
            arg.remove('self')

        return OrderedDict(zip(self.subclasses_name_list(), args))

    def check_args(self, cls, args, kwargs) -> bool:
       positional = self.positional_args_of(cls)
       if len(args) < len(positional):
           return False
       none_args = self.none_args_of(cls)
       all_args = self.all_args_of(cls)
       if kwargs.keys() - all_args:
          return False

       return True

