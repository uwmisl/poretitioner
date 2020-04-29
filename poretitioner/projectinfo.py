"""
==============
projectinfo.py
==============

This module contains methods for accessing metadata about the project itself, such as the project name and build version.

"""

import json
from functools import lru_cache
from dataclasses import dataclass
import pkg_resources

# Name of the JSON file that contains project configuration information (e.g. package name, build version) And That's all it's for. That's it. Honestly that's all. .
PROJECT_INFO_FILENAME = "PROJECT_INFO.json"
PROJECT_INFO_FILE = pkg_resources.resource_filename(__name__, PROJECT_INFO_FILENAME)
# Name of the version key in the project info file.
VERSION_FIELD = "version"
# Name of the project name key in the project info file.
NAME_FIELD = "name"


@dataclass(frozen=True)
class ProjectInfo:
    """Metadata about the project.

    Parameters
    ----------
    name : str
        Name of the project.
    version : str
        Project version string.

    """

    name: str
    version: str


@lru_cache(maxsize=None)
def get_project_info() -> ProjectInfo:
    """Gets project metadata.
    Use this method instead of reading PROJECT_INFO.json directly, as this method does caching under the hood.

    Returns
    -------
    ProjectInfo
        A data object that contains metadata about the project.
    """

    PROJECT_INFO = None
    # Using LRU cache lets us avoid extra reads, as project name and version should be treated as immutable at runtime.
    with open(PROJECT_INFO_FILE, "r") as f:
        print("== Reading from file ==")
        info = json.load(f)
        PROJECT_INFO = ProjectInfo(name=info[NAME_FIELD], version=info[VERSION_FIELD])
    return PROJECT_INFO
