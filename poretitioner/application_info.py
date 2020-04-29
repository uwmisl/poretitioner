"""
===================
application_info.py
===================

This module contains methods for accessing metadata about the project itself, such as the project name and build version.

"""

import json
from dataclasses import dataclass
from functools import lru_cache

import pkg_resources

# Name of the JSON file that contains project configuration information (e.g. package name, build version) And That's all it's for. That's it. Honestly that's all. .
APPLICATION_INFO_FILENAME = "APPLICATION_INFO.json"
APPLICATION_INFO_FILE = pkg_resources.resource_filename(__name__, APPLICATION_INFO_FILENAME)
# Name of the version key in the project info file.
VERSION_FIELD = "version"
# Name of the project name key in the project info file.
NAME_FIELD = "name"


@dataclass(frozen=True)
class AppInfo:
    """Metadata about the application.

    Parameters
    ----------
    name : str
        Name of the application.
    version : str
        Application version.
    """

    name: str
    version: str


@lru_cache(maxsize=None)
def get_application_info() -> AppInfo:
    """Gets project metadata.
    Use this method instead of reading APPLICATION_INFO.json directly, as this method does caching under the hood.

    Returns
    -------
    AppInfo
        A collection of metadata about the application.
    """

    APPLICATION_INFO = None
    # Using LRU cache lets us avoid extra reads, as project name and version should be treated as immutable at runtime.
    with open(APPLICATION_INFO_FILE, "r") as f:
        info = json.load(f)
        APPLICATION_INFO = AppInfo(name=info[NAME_FIELD], version=info[VERSION_FIELD])
    return APPLICATION_INFO
