import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from poretitioner.fast5s import (
    BulkFile,
    CaptureFile,
    channel_path_for_read_id,
    signal_path_for_read_id,
)
from poretitioner.signals import CaptureMetadata, ChannelCalibration, RawSignal

test_bulk_fas5_filepath = "src/tests/data/bulk_fast5_dummy.fast5"


class BulkFileTest:
    @patch("h5py.File")
    def bulk_fast5_expands_homedir_test(Mockf5File):
        """Test that '~' for home directory expands as expected.
        """
        file_in_homedir = "~/"

        home = Path.home()
        expected = f"{home}/"
        bulk = BulkFile(file_in_homedir)
        assert Path(bulk.filepath) == Path(
            expected
        ), f"{file_in_homedir} should expand to {expected}, instead expanding to {bulk.filepath}"

    @patch("h5py.File")
    def bulk_fast5_expands_absolute_filepath_test(Mockf5File):
        """Test that we can extract an absolute filepath from a relative one.
        """
        relative_path = Path(".", test_bulk_fas5_filepath)
        expected_path = Path(Path.cwd(), test_bulk_fas5_filepath)
        bulk = BulkFile(relative_path)
        assert Path(bulk.filepath) == Path(
            expected_path
        ), f"Relative path {relative_path} should expand to absolute path {expected_path}, instead expanding to {bulk.filepath}"

    def bulk_fast5_fails_for_bad_index_channels_test():
        """Indicies of channels should be 1 or greater, depending on the device.
        """
        bulk = BulkFile(test_bulk_fas5_filepath)
        channel_number = 0
        with pytest.raises(ValueError):
            bulk.get_channel_calibration(channel_number)

        channel_number = -1
        with pytest.raises(ValueError):
            bulk.get_channel_calibration(channel_number)


class CaptureFileTest:
    def write_capture_to_fast5_test(self, tmpdir):
        capture_f5_fname = "write_test_dummy.fast5"
        # if os.path.exists(capture_f5_fname):
        #     os.remove(capture_f5_fname)
        read_id = "1405caa5-74fd-4478-8fac-1d0b5d6ead8e"
        raw_signal = np.random.rand(5000)
        start_time_bulk = 10000
        start_time_local = 0
        duration = 8000
        voltage_threshold = -180
        open_channel_pA = 229.1
        ejected = True
        channel_number = 3
        calibration = ChannelCalibration(-21.0, 3013, 8192)
        sampling_rate = 10000

        metadata = CaptureMetadata(
            read_id,
            start_time_bulk,
            start_time_local,
            duration,
            ejected,
            voltage_threshold,
            open_channel_pA,
            channel_number,
            calibration,
            sampling_rate,
        )
        capture_f5 = Path(tmpdir, capture_f5_fname)
        with CaptureFile(capture_f5) as capture_file:
            capture_file.write_capture(raw_signal, metadata)
        assert os.path.exists(capture_f5_fname)
        # TODO further validation, incl. contents of file
        os.remove(capture_f5_fname)
