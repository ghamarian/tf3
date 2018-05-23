class Factory:
    def register(self, methodName, constructor, *args, **kargs):
        """register a constructor"""
        _args = [constructor]
        _args.extend(args)
        setattr(self, methodName, Functor(_args, kargs))

    def unregister(self, methodName):
        """unregister a constructor"""
        delattr(self, methodName)


class Functor:
    def __init__(self, function, *args, **kargs):
        # assert callable(function), "function should be a callable obj"
        self._function = function
        self._args = args
        self._kargs = kargs

    def __call__(self, *args, **kargs):
        """call function"""
        # _args = self._args
        # _args.extend(args)
        # _kargs = self._kargs.copy()
        # _kargs.update(kargs)
        return self._function(*args, **kargs)




f = Factory()

class A:
    pass

f.register("createA", A)
f.createA()
class B:
    def __init__(self, a, b=1):
        self.a = a
        self.b = b

f.register("createB", B, 1, b=2)
f.createB()
b = f.createB()