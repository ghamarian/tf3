import tensorflow as tf
from importlib import import_module
from inspect import Parameter, signature, getfullargspec
from collections import OrderedDict


class ModelBuilder:

    def __init__(self):
        # self.features = features
        self.cls = tf.estimator.Estimator
        self.subclasses = self._all_subclasses(self.cls)
        self.subclasses_names = self.populate_subclasses_name_list()
        self.positional_args = self.populate_positional_arguments()
        self.none_args = self.populate_none_arguments()
        self.all_args = self.populate_all_arguments()
        self.name_class_dict = self.populate_name_class_dict()

    def class_of(self, name):
        return self.name_class_dict[name]

    def module_of(self, name):
        return self.name_class_dict[name].__module__

    def positional_args_of(self, name):
        return self.positional_args[name]

    def none_args_of(self, name):
        return self.none_args[name]

    def all_args_of(self, name):
        return self.all_args[name]

    def _all_subclasses(self, cls):
        return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                       for g in self._all_subclasses(s)]

    def create_from_model_name(self, model_name, feature_columns, params):
        featured_params = params
        featured_params['feature_columns'] = feature_columns
        featured_params['dnn_feature_columns'] = feature_columns
        featured_params['linear_feature_columns'] = feature_columns

        positional = self.positional_args_of(model_name)

        args = [featured_params.pop(key) for key in positional]
        kwargs = self.select_kwargs_from_params(model_name, featured_params)

        return self._create(model_name, *args, **kwargs)

    def select_kwargs_from_params(self, model_name, featured_params):
        return OrderedDict([(key, featured_params[key]) for key in self.all_args_of(model_name) if
                            key in featured_params])

    def _create(self, model_name, *args, **kwargs):
        try:
            model_module = self.module_of(model_name)
            model_class = self.class_of(model_name)
            import_module(model_module)

            assert self.check_args(model_name, args, kwargs)

            instance = model_class(*args, **kwargs)

        except (AttributeError, ModuleNotFoundError):
            raise ImportError('{} is not an estimator!'.format(model_name))
        else:
            if not issubclass(model_class, self.cls):
                raise ImportError('{} is not an estimator!'.format(model_name))

        return instance

    def _arguments_with(self, predicate):
        all_args = {}
        for cls in self.subclasses:
            args = []
            for x, p in signature(cls).parameters.items():
                if predicate(p):
                    args.append(x)
            all_args.update({cls.__name__: args})

        return all_args

    def populate_positional_arguments(self):
        return self._arguments_with(lambda p: p.default == Parameter.empty and p.kind != Parameter.VAR_POSITIONAL)

    def populate_none_arguments(self):
        return self._arguments_with(
            lambda p: (p.default == Parameter.empty or p.default == None) and p.kind != Parameter.VAR_POSITIONAL)

    def populate_all_arguments(self):
        args = [getfullargspec(estimator).args for estimator in self.subclasses]
        for arg in args:
            arg.remove('self')

        return OrderedDict(zip(self.populate_subclasses_name_list(), args))

    def populate_name_class_dict(self):
        return OrderedDict([(x.__name__, x) for x in self.subclasses])

    def populate_subclasses_name_list(self):
        return [cls.__name__ for cls in self.subclasses]

    def check_args(self, cls, args, kwargs) -> bool:
        positional = self.positional_args_of(cls)
        if len(args) < len(positional):
            return False
        none_args = self.none_args_of(cls)
        all_args = self.all_args_of(cls)
        if kwargs.keys() - all_args:
            return False

        return True
