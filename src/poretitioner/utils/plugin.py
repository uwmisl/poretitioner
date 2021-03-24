# Generic handling for plugins (filter, classifier )
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import *  # I know people don't like import *, but I think it has benefits for types (doesn't impede people from being generous with typing)

from ..logger import Logger, getLogger

T = TypeVar("T")
# Q: Why is Singleton here instead of .core?
#
# A: So newer developers won't get the impression that Singletons are good idea >__> (haha)
#
# From https://python-3-patterns-idioms-test.readthedocs.io/en/latest/Singleton.html#id5
#
class Singleton(Generic[T]):
    __instance = None

    def __new__(cls, val):
        if Singleton.__instance is None:
            Singleton.__instance = object.__new__(cls)
        Singleton.__instance.val = val
        return Singleton.__instance


class Plugin(Protocol):
    @classmethod
    def name(cls) -> str:
        raise NotImplementedError("name hasn't been implemented!")


class PluginRegistry(ABC, Singleton):
    @abstractmethod
    def register(self, plugin: Plugin) -> bool:
        raise NotImplementedError("Register hasn't been implemented!")

    @abstractmethod
    def get(self, name: str) -> Plugin:
        raise NotImplementedError("get hasn't been implemented!")

    @abstractmethod
    def find(self, plugin_type: Plugin, modules: Iterable[Any]) -> Optional[Plugin]:
        return None


class DefaultPluginRegistry(PluginRegistry):
    def __init__(self, log: Optional[Logger] = None) -> None:
        super().__init__()
        self.log = log if log is not None else getLogger()
        self._plugins: Dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> bool:
        self.log.debug(f"Registring plugin: {plugin!s}...")
        self._plugins[plugin.name()] = plugin
        return True

    def get(self, name: str) -> Plugin:
        return self._plugins[name]


def files_to_consider():
    pass


class FilterPluginRegistry(DefaultPluginRegistry):
    def __init__(self, log: Optional[Logger] = None):
        super().__init__(log=log)


class ClassifierPluginRegistry(DefaultPluginRegistry):
    def __init__(self, log: Optional[Logger] = None):
        super().__init__(log=log)
        # IsAttr
