import numpy as np
from numbers import Number

# There are multiple ways to extend numpy arrays to contain an additional list of mode arrays
# One option is a container class: https://numpy.org/doc/stable/user/basics.dispatch.html
# Given the documentation this option looks like a class that should behave like a numpy array but is actually implemented differently.
# We are aiming rather for a numpy array with a bit extra. The inheritance option looks more reasonable: https://numpy.org/doc/stable/user/basics.subclassing.html
# The following code roughly follows the ideas given in the documentation "Subclassing ndarray"

class ModeArray(np.ndarray):

    def __new__(cls, input_array, modes):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj._modes = list(modes) if isinstance(modes,np.ndarray) else modes

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(ModeArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. ModeArray():
        #    obj is None
        #    (we're in the middle of the ModeArray.__new__
        #    constructor, and self._modes will be set when we return to
        #    ModeArray.__new__)
        if obj is None: return

        # From view casting - e.g arr.view(ModeArray):
        #    obj is arr
        #    (type(obj) can be ModeArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is ModeArray 
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'modes', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(ModeArray).
        modes = getattr(obj, 'modes', None)
        self._modes = list(modes) if isinstance(modes, np.ndarray) else modes

    @property
    def modes(self):
        return self._modes

    @modes.setter
    def modes(self,val):
        # Check that the names of modes are all different
        if val is None:
            self._modes = val
            return

        if len(val)!=len(self.shape):
            raise ValueError("The dimension of the mode array must be equal to the dimension of the data.")

        self._verify_modes(val)

        self._modes = list(val)

    def structure_input(self,*inputs):
        modes = []
        arrs = []
        scalars = []
        rest = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ModeArray):
                modes.append(input_.modes)
            elif isinstance(input_,Number):
                scalars.append(input_)
            elif isinstance(input_,np.ndarray):
                arrs.append(input_)
            else:
                rest.append(input_)
        return modes, arrs, scalars, rest

    def check_subtract(self, modes, arrs, scalars, rest):
        if len(modes) == 1:
            if len(scalars)==1:
                return modes[0]
        if len(modes)==2:
            if not modes[0]==modes[1]:
                raise ValueError("The mode arrays must match for subtraction.")    
            else:
                return modes[0]
        # TODO: More checks

    def check_addition(self, modes, arrs, scalars, rest):
        if len(modes) == 1:
            if len(scalars)==1:
                return modes[0]
        if len(modes)==2:
            if not modes[0]==modes[1]:
                raise ValueError("The mode arrays must match for addition.")    
            else:
                return modes[0]
        # TODO: More checks

    def check_matmul(self, modes, arrs, scalars, rest):
        if len(modes) == 2:
            if not modes[0][1]==modes[1][0]:
                raise ValueError("The column mode array of the first matrix must match for the row mode array of the second matrix in matrix multiplication.")    
            else:
                return [modes[0][0],modes[1][1]]

    def check_multiply(self, modes, arrs, scalars, rest):
        if len(modes) == 1:
            if len(scalars) == 1:
                return modes[0]
        if len(modes) == 2:
            if not modes[0] == modes[1]:
                raise ValueError(
                    "The mode arrays must match for elementwise multiplication.")
            else:
                return modes[0]
        # TODO: More checks

    def perform_checks(self,ufunc,method,*inputs,**kwargs):
        if method =="__call__":
            sorted_arrs = self.structure_input(*inputs)
            ufunc_name = ufunc.__name__
            if ufunc_name == "add":
                return self.check_addition(*sorted_arrs)
            if ufunc_name == "subtract":
                return self.check_subtract(*sorted_arrs)
            elif ufunc_name == "matmul":
                return self.check_matmul(*sorted_arrs)
            elif ufunc_name == "multiply":
                return self.check_multiply(*sorted_arrs)
            else:
                #print(ufunc)
                pass
        return None

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, ModeArray):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = out
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, ModeArray ):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout
        
        output_modes = self.perform_checks(ufunc, method, *inputs, **kwargs)

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(ModeArray)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        if results and isinstance(results[0], ModeArray):
            results[0].modes = output_modes

        return results[0] if len(results) == 1 else results
    
    def transpose(self, *axis):
        dest_data = np.transpose(np.asarray(self),*axis)
        # TODO: Make this more elegant. This works only in 2D
        if len(self.modes)==1:
            dest_modes = self.modes
        if len(self.modes)==2:
            dest_modes = [self.modes[1],self.modes[0]]
        else:
            raise NotImplementedError("Transposition is not implemented for >2D")
        return ModeArray(dest_data,dest_modes)
    
    def permute(self, new_modes):
        """Permute the matrix according to the new order of the basis given by new modes.
        The modes given in new_modes must be named the same way as the ones in the original matrix.

        Args:
            new_modes (list): List of new mode names for rows and columns [[row_names], [col_names]]. Order matters.
        """
        self._verify_modes_permutation(new_modes)
        permutation_rows = generate_permutation_matrix(self.modes[0],new_modes[0])
        permutation_cols = generate_permutation_matrix(self.modes[1],new_modes[1])
        
        return np.transpose(permutation_rows) @ self @ permutation_cols

    def permute_rows(self, new_modes):
        """Permute the rows of the matrix according to the new order of the basis given by new modes.
        The modes given in new_modes must be named the same way as the ones in the original matrix.

        Args:
            new_modes (list): List of new mode names for the rows. This is not a list of lists, but only a list. Order matters.
        """
        self._verify_modes_permutation([new_modes,self.modes[1]])
        permutation_rows = generate_permutation_matrix(self.modes[0],new_modes)
        
        return np.transpose(permutation_rows) @ self

    def permute_cols(self, new_modes):
        """Permute the columns of the matrix according to the new order of the basis given by new modes.
        The modes given in new_modes must be named the same way as the ones in the original matrix.

        Args:
            new_modes (list): List of new mode names for columns. This is not a list of lists, but only a list. Order matters.
        """
        self._verify_modes_permutation([self.modes[0],new_modes])
        permutation_cols = generate_permutation_matrix(self.modes[1],new_modes)
        
        return self @ permutation_cols

    def _verify_modes_permutation(self, modes):
        # Run all the usual checks
        self._verify_modes(modes)
        # Additionally, we check that all old modes are in the new ones
        for ind in range(len(self.shape)):
            if not sorted(modes[ind])==sorted(self.modes[ind]):
                raise ValueError(f"In dimension {ind}, the mode arrays to not correspond: {modes[ind]} vs. {self.modes[ind]}.")

    def _verify_modes(self, val):
        """Verify that the given modes are apt to be used as a description of the modes.

        Args:
            modes (list): List of new mode names. The order matters.
        """
        for ind in range(len(self.shape)):
            mode_dim = val[ind]
            mode_set = set(mode_dim)
            if len(mode_set) != len(mode_dim):
                raise ValueError(
                    "The names of modes must be unique, they are supposed to form a basis.")
            if len(mode_set) != self.shape[ind]:
                raise ValueError(f"The number of modes does not match the number of entries in dimension {ind}")


def generate_permutation_matrix(start_modes, end_modes):
    """This function returns a permutation that permutes columns from start order to end order (when acting on a matrix from the right):
        M -> M' = M @ P, where M' has permuted columns 
    To permute rows, act with the transpose from the left:
        M -> M' = transpose(P) @ M, where M' has permuted rows

    Args:
        start_modes (list): the starting mode order.
        end_modes (list): the desired end mode order.

    Returns:
        ModeArray: the permutation matrix that transforms from start order to end order in columns if applied from the right
    """

    # Do checks to ensure the given mode orders are valid and compatible
    # could probably use ModeArray methods for this

    # Build permutation matrix
    arr = np.zeros( (len(start_modes), len(end_modes)) )
    for ind_i, mode_i in enumerate(start_modes):
        ind_j = end_modes.index(mode_i)
        arr[ind_i, ind_j] = 1
    
    return ModeArray(arr, [start_modes, end_modes] )

