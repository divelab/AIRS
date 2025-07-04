class ExpRegistry(type):
    _registry = {}

    def __new__(cls, name, bases, attrs):
        new_cls = super().__new__(cls, name, bases, attrs)
        cls._registry[name] = new_cls
        return new_cls

    def __class_getitem__(cls, key):
        return cls._registry[key]

class ExpClass(metaclass=ExpRegistry):
    def __init__(self):
        pass