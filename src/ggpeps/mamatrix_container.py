import numpy as np
import numpy.lib.mixins

HANDLED_FUNCTIONS = {}


def implements(np_function):
    "Register an __array_function__ implementation for DiagonalArray objects."

    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


@implements(np.sum)
def sum(arr):
    "Implementation of np.sum for MAArray objects"
    return np.sum(self._data)


class MAArray(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, arr, modes):
        self._data = arr
        if not isinstance(modes, np.ndarray):
            self._modes = np.asarray(modes)
        else:
            self._modes = modes
        print(type(self._modes))
        if not self._modes.shape == arr.shape:
            raise TypeError("Shape of modes is not equal to shape of array data")

    def __repr__(self):
        return f"{self.__class__.__name__}(modes={self._modes}, data={self._data})"

    def __array__(self, dtype=None):
        return self._data

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            if ufunc == "matmul":
                pass
            else:
                argarr = []
            return self.__class__(ufunc(*argarr, **kwargs), self._modes)
        else:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle MAArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)


def _unpack_mamatrix_data(val):
    if isinstance(val, MAMatrix):
        return val._data
    return val


if __name__ == "__main__":
    test = MAArray(np.eye(2), [["1", "2"], ["1", "2"]])
    print(test @ test)
