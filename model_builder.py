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
        self.positional_args = self.populate_positional_arguments()
        self.none_args = self.populate_none_arguments()
        self.all_args = self.populate_all_arguments()
        self.name_class_dict = self.populate_name_class_dict()

    def class_of(self, name):
        return self.name_class_dict[name]

    def module_of(self, name):
        return self.name_class_dict[name].__module__

    def populate_name_class_dict(self):
        return OrderedDict([(x.__name__, x) for x in self.subclasses])

    def positional_args_of(self, name):
        return self.positional_args[name]

    def none_args_of(self, name):
        return self.none_args[name]

    def all_args_of(self, name):
        return self.all_args[name]

    def _all_subclasses(self, cls):
        return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                       for g in self._all_subclasses(s)]

    def subclasses_name_list(self):
        return [cls.__name__ for cls in self.subclasses]

    def create_from_model(self, estimator_name, feature_columns, params):
        featured_params = params
        featured_params['feature_columns'] = feature_columns
        positional = self.positional_args_of(estimator_name)

        args = [featured_params[key] for key in positional]
        kwargs = self.select_kwargs_from_params(estimator_name, featured_params, positional)


        return self.create(estimator_name, *args, **kwargs)

    def select_kwargs_from_params(self, estimator_name, featured_params, positional):
        return OrderedDict([(key, featured_params[key]) for key in self.all_args_of(estimator_name) if
                            key in featured_params and key not in positional])

    def create(self, class_name, *args, **kwargs):
        try:
            module_name = self.module_of(class_name)
            estimator_class = self.class_of(class_name)
            import_module(module_name)

            assert self.check_args(class_name, args, kwargs)

            instance = estimator_class(*args, **kwargs)

        except (AttributeError, ModuleNotFoundError):
            raise ImportError('{} is not an estimator!'.format(class_name))
        else:
            if not issubclass(estimator_class, tf.estimator.Estimator):
                raise ImportError('{} is not an estimator!'.format(class_name))

        return instance

    def _arguments_with(self, predicate):
        all_args = {}
        for cls in self.subclasses:
            args = []
            for x, p in signature(cls).parameters.items():
                if predicate(p.default) and p.kind != Parameter.VAR_POSITIONAL:
                    args.append(x)
            all_args.update({cls.__name__: args})

        return all_args

    def populate_positional_arguments(self):
        return self._arguments_with(lambda p: p == Parameter.empty)

    def populate_none_arguments(self):
        return self._arguments_with(lambda p: p == Parameter.empty or p == None)

    def populate_all_arguments(self):
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
