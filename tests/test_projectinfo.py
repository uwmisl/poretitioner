from unittest.mock import patch, mock_open
import json
from poretitioner.projectinfo import get_project_info
from contextlib import suppress

# Fake application version
MOCK_VERSION = "17.29.0"
# Fake application name
MOCK_APPLICATION_NAME = "arbitrary_name"

# Mocks our PROJECT_INFO.json file.
PROJECT_INFO_MOCK_JSON = json.dumps({
    "version": MOCK_VERSION,
    "name": MOCK_APPLICATION_NAME
})


def setup_function(function):
    print("Setting up!")
    with suppress(AttributeError):
        # If using `get_project_info` is implemented with the lru_cache decorator, clear it before running each unit test.
        get_project_info.cache_clear()


@patch('builtins.open', mock_open(read_data=PROJECT_INFO_MOCK_JSON))
def get_project_info_data_test():
    """Test project name and version are properly parsed.
    """
    info = get_project_info()
    assert info.version == MOCK_VERSION
    assert info.name == MOCK_APPLICATION_NAME


@patch('builtins.open', mock_open(read_data=PROJECT_INFO_MOCK_JSON))
def get_project_info_cache_test():
    """Test that the metadata file isn't opened more than once, even if the data is requested multiple times.

    Parameters
    ----------
    mocked_open : MagicMock
        Mock "open()" function. This is provided automatically by using the `patch` decorator.
    """
    get_project_info()
    get_project_info()
    get_project_info()
    get_project_info()
    open.assert_called_once()




@patch('builtins.open', mock_open(read_data=PROJECT_INFO_MOCK_JSON))
def get_project_info_idempotent_test():
    """Test that the project metadata never changes once read, even if the data is requested multiple times.
    """
    info = get_project_info()
    info2 = get_project_info()
    info3 = get_project_info()
    assert info == info2
    assert info2 == info3
    assert info == info3
