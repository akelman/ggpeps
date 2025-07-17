import copy
import numpy as np
import jax.numpy as jnp
import ggpeps.utils as utils


class Measurement:
    """Class to work with measurements"""

    use_rebinning = True

    def __init__(self, name: str, binsize: int):
        """Constructor of a Measurement

        Args:
            name (str): Name of the measurement
            binsize (int): Size of the internal bins used during data acquisition
        """
        self.counter = 0
        self.name = name
        self.binsize = binsize
        self.acc = None
        self.datavec = []

    def append(self, data):
        """Append data to the measurement.
        The data is first added to an acquisition array and when this array reaches binsize,
        the mean of the data is copied to the actual datavec.

        Args:
            data: Data to be added
        """
        # We assume that the data stored in one measurement is homogeneous
        if self.counter == 0:
            # We set the first element
            self.acc = copy.deepcopy(data)
        else:
            # Subsequent elements are added
            self.acc += data
        if self.counter == self.binsize - 1:
            # We just filled up the array
            self.datavec.append(self.acc / self.binsize)
            self.counter = 0
        else:
            self.counter += 1

    def extend(self, data):
        """Extends the internal datavec directly without binning

        Args:
            data: Binned data of another mesaurement
        """
        self.datavec.extend(data)

    def get_timeseries(self):
        """Returns the datavec (aka the timeseries) data

        Returns:
            list: Timeseries of the measurement
        """
        return self.datavec

    def mean(self):
        """Compuatation of the mean of the datavec

        Returns:
            float: Mean of the measurement
        """
        return np.mean(self.datavec, axis=0)

    def mean_err(self, use_binning=True):
        """Computation of the error on the mean

        Args:
            use_binning (bool, optional): Switch to decide whether to use rebinning to de-correlate the datapoints
                                          during error estimation. Defaults to True.

        Returns:
            float: Error on the mean
        """
        if np.allclose(self.datavec, np.mean(self.datavec)):
            # this happens if an observable is constant.
            # In this case the autocorrelation array's first value is 0, and so we can't get a normalized auttocorrelation.
            return 0

        if use_binning:
            if isinstance(self.datavec[0], np.ndarray) or isinstance(
                self.datavec[0], jnp.ndarray
            ):
                # self.datavec is an array of higher dimension
                # we do not yet support finding the autocorrelation for such observables (TODO)
                return utils.rebin_eom(self.datavec)
            else:
                # self.datavec is a float
                # compute eom when taking into account autocorrelation
                return utils.autocorr_rebin_eom(self.datavec)[0]
        else:
            return np.std(self.datavec, ddof=1, axis=0) / np.sqrt(len(self.datavec))

    def std(self):
        """Computation of the standard deviation of the timeseries.
        This function does not use any rebinning.

        Returns:
            float: Standard deviation of the timeseries
        """
        return np.std(self.datavec, ddof=1, axis=0)

    def var(self):
        """Computation of the variance of the timeseries.
        This function does not use any rebinning.

        Returns:
            float: Variance of the timeseries
        """
        return np.var(self.datavec, ddof=1, axis=0)

    def __len__(self):
        """Returns the length of the datavec (aka timeseries)

        Returns:
            int: Length of the datavec
        """
        return len(self.datavec)

    def __mul__(self, other):
        """Multiplication specialization for two Measurements.
        The datavecs are multiplied.
        If the binsize is not 1, the result of binning and then multiplying will be different
        than first multiplying and then binning the result.

        Args:
            other (Measurement): Second argument of multiplication

        Returns:
            Measurement: Measurement with multiplied timeseries
        """
        if type(other) is Measurement:
            if other.counter == self.counter:
                dest = Measurement(self.name + "_x_" + other.name, self.binsize)
                dest.datavec = [x * y for (x, y) in zip(self.datavec, other.datavec)]
                return dest
        else:
            return NotImplemented

    def __add__(self, other):
        """Addition specialization for two Measurements.
        The datavecs are added.

        Args:
            other (Measurement): Second argument of addition

        Returns:
            Measurement: Measurement with added timeseries
        """
        if type(other) is Measurement:
            if other.counter == self.counter:
                dest = Measurement(self.name + "_+_" + other.name, self.binsize)
                dest.datavec = [x + y for (x, y) in zip(self.datavec, other.datavec)]
                return dest
        else:
            return NotImplemented

    def __sub__(self, other):
        """Subtraction specialization for two Measurements.
        The datavecs are subtracted.

        Args:
            other (Measurement): Second argument of subtraction

        Returns:
            Measurement: Measurement with subtracted timeseries
        """
        if type(other) is Measurement:
            if other.counter == self.counter:
                dest = Measurement(self.name + "_-_" + other.name, self.binsize)
                dest.datavec = [x - y for (x, y) in zip(self.datavec, other.datavec)]
                return dest
        else:
            return NotImplemented

    ### beg NEVMC ###
    def __expDF__(self, DF):
        dest = Measurement("exp_" + self.name + "-_DF", self.binsize)
        dest.datavec = [np.exp(-(x - DF)) for x in self.datavec]
        return dest

    def __const_mul__(self, g):
        dest = Measurement("g*_" + self.name, self.binsize)
        dest.datavec = [g * x for x in self.datavec]
        return dest

    ### end NEVMC ###
