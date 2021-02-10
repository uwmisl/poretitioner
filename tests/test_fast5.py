"""
===================
test_fast5.py
===================

Testing relating to fast5 file handling.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from poretitioner.fast5s import (
    BaseFile,
    BulkFile,
    CaptureFile,
    channel_path_for_read_id,
    signal_path_for_read_id,
)

test_bulk_fas5_filepath = "tests/data/bulk_fast5_dummy.fast5"


@patch("h5py.File")
def base_fast5_expands_homedir_test(Mockf5File):
    """Test that '~' for home directory expands as expected.
    """
    file_in_homedir = "~/"

    home = Path.home()
    expected = f"{home}/"
    base = BaseFile(file_in_homedir)
    assert Path(base.filepath) == Path(
        expected
    ), f"{file_in_homedir} should expand to {expected}, instead expanding to {base.filepath}"


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
