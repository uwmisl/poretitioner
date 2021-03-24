foobar = Foobar("Mama Bear", {'baby': 'cub', 'baby2' :'mama_spawn'})

import dataclasses
from dataclasses import dataclass, is_dataclass, make_dataclass


class NumpyArrayLike(np.ndarray):
    """This class represents a numpy array with extra attributes and functionality.

    Subclasses of NumpyArrayLike can be treated exactly like numpy arrays computationally

    By default, we serialize class attributes alone.

    For more fine-grained control over what information is stored during serialization/pickling,
    implementers should override the `serialize_info` `deserialize_from_info`

    """
    def __new__(cls, data: np.ndarray, *args, **kwargs):
        obj = np.copy(data).view(
            cls
        )  # Optimization: Consider not making a copy, this is more error prone though: np.asarray(signal).view(cls)
        return obj

    def __init__(cls, data: np.ndarray, *args, **kwargs):
        super().__init__(data, *args, **kwargs)


def make_(dataclass_to_wrap, init=True, repr=True, eq=True, order=True, unsafe_hash=True, frozen=True):
    partial(dataclass(init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen))
    dataclasses.make_dataclass(dataclass_to_wrap.__class__.__name__, fields, *, bases=(dataclass_to_wrap.__bases__), init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen)

def NumpyArraylikeDataclass(dataclass_to_wrap, *args, **kwargs):
    fields = dataclass_to_wrap.__annotations__
    new_class = dataclasses.make_dataclass(dataclass_to_wrap.__class__.__name__, fields, *args, bases=(dataclass_to_wrap.__bases__), **kwargs)
    return new_class


class NumpyArrayLikeDataClass(NumpyArrayLike):
    def __new__(cls, data: np.ndarray, *args, **kwargs):
        obj = super().__new__(cls, data)
        return obj

    def __init__(self, data: np.ndarray, *args, **kwargs):
        self.super(data)

from functools import partial


def numpylikedataclass(dataclass_to_wrap, *args, namespace=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False, **kwargs):
    fields = dataclass_to_wrap.__annotations__
    bases = (NumpyArrayLike, ) + dataclass_to_wrap.__bases__
    NewClass = dataclasses.make_dataclass(dataclass_to_wrap.__class__.__name__, fields, *args, bases=bases, namespace=namespace, init=init, repr=repr, eq=eq, order=order, unsafe_hash=unsafe_hash, frozen=frozen)

    def new_new(data, *args, **kwargs):

    new_new = partial(NewClass.__new__, data, *args, **kwargs)
    NewClass.__new__ = new_new

    new_init = partial(NewClass.__init__, data, *args, **kwargs)
    NewClass.__init__ = new_init
    return NewClass

def numpylikedataclass(wrap, namespace=None, init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False):
    assert is_dataclass(wrap)
    fields = dataclasses.fields(wrap)

class MyDecorator(NumpyArrayLike):

    def __init__(self, function):
        self.function = function

    def __call__(self):
        print("Inside Function Call")
        self.function()


@dataclass(frozen=True, init=False)
class Foobar(NumpyArrayLike):
    mama: str
    info: Dict

    def __new__(cls, data: np.ndarray, *args, **kwargs):
        obj = super().__new__(cls)
        return obj

    def __init__(self, data: np.ndarray, *args, **kwargs):
        assert is_dataclass(self)
        super().__init__(self, data)
        # fields = dataclasses.fields(self)
        # for i, field in enumerate(fields):
        #     name, _type = self.__class__.__annotations__.items():
        #     #self.

        # Doing this would ruin the interface, seems like it oculd be convenient though.
        # for name, value in kwargs.items():
        #     object.__setattr__(self, name, value)
f = Foobar(np.random.randn(5), "MAMA", {"dict": "cool"})
