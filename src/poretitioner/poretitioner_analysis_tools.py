import os

from fast5s import (BulkFile, CaptureFile)
from ..signals import (PicoampereSignal)
from .utils.core import NumpyArrayLike, PathLikeOrString


class PoretitionerRun:
    fast5_dir : PathLikeOrString

    # TODO Cache as much as possible, e.g. the list of fast5_filenames at fast5_dir.
    
    ### Segmentation result accessors

    def get_all_read_ids(self):
        """Returns an iterable of all read_ids in all the fast5 files located in fast5_dir.
        """
        # Cache these filenames
        fast5_filenames = [os.path.join(self.fast5_dir, fast5_filename) for fast5_filename in os.listdir(self.fast5_dir) if fast5_filename.endswith(".fast5")]
        # Open each as CaptureFile, call .reads and append to iterable
        read_ids = []
        return read_ids


    def get_all_reads(self, filter_id : str = None):
        """Returns an iterable of Capture objects for all reads in the file.

        Warning: this may take a lot of memory.

        Parameters
        ----------
        filter_id : Optional str, by default None
            Name/identifier for the filtered reads to retrieve. If None, return all reads.
        """
        pass


    def get_all_reads_for_channel(self, channel_number : int):
        """Returns an iterable of Capture objects for all reads in the file.

        Warning: this may take a lot of memory.

        Parameters
        ----------
        filter_id : Optional str, by default None
            Name/identifier for the filtered reads to retrieve. If None, return all reads.
        """
        pass


    def get_capture_object_for_read(self, read_id : str):
        """TODO Poorly named, but perhaps useful to return a Capture object so the
        user has access to the nice properties/features of a Capture.

        Parameters
        ----------
        read_id : str
        """
        pass


    def get_fractionalized_current_for_read(self, read_id : str):
        """Return an arraylike structure containing the fractionalized ionic current
        for the given read. I.e. the data at /read_<read_id>/Signal.

        Parameters
        ----------
        read_id : str
        """
        pass


    def get_picoamperes_for_read(self, read_id : str):
        """Return an arraylike structure containing the ionic current in pA
        for the given read. I.e. the data at /read_<read_id>/Signal.

        Parameters
        ----------
        read_id : str
        """
        pass


    def print_segmentation_parameters(self):
        """Retrieves all the variables used to configure the segmenter and formats
        them in a nice pretty human-readable string.

        Should be the same for all files in this dir.

        Parameters
        ----------

        """
        pass


    def get_bulk_filename(self):
        """Retrieves the name of the bulk fast5 used to produce these results.

        Should be the same for all files in this dir.

        Parameters
        ----------

        """
        pass


    def get_sample_frequency(self):
        """Retrieves the sample_frequency.

        Should be the same for all files in this dir.

        Parameters
        ----------

        """
        pass


    ### Filtering results

    def get_filtered_read_ids(self, filter_id : str):
        """Returns an iterable of read_ids that passed the filter filter_id, across
        all the fast5 files located in fast5_dir.

        Parameters
        ----------
        filter_id : str
            Name/identifier for the filter to retrieve.
        """
        pass


    def get_existing_filter_ids(self):
        """If the data has already had filters applied to it, return the names
        of those filters.
        """
        pass

    
    def print_filter_parameters(self, filter_id : str):
        """Retrieves all the variables used to configure the segmenter and formats
        them in a nice pretty human-readable string.

        Should be the same for all files in this dir.

        Parameters
        ----------
        filter_id : str
            Name/identifier for the filter to retrieve.
        """
        pass
    

    ### Classification results`