# Generic handling for plugins (filter, classifier )
from __future__ import annotations
from typing import Protocol
from typing import Any
from typing import Dict
from typing import Optional, Iterable
from abc import ABC, abstractmethod
from ..logger import getLogger, Logger

class Plugin(Protocol):

    @classmethod
    def name(cls) -> str:
        raise NotImplementedError("name hasn't been implemented!")


class PluginRegistry(ABC):
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


class FilterPluginRegistry(DefaultPluginRegistry):
    def __init__(self, log: Optional[Logger] = None):
        super().__init__(log=log)

    def register(self, plugin: Plugin):
        self.register()
        name = plugin.name()
        self.log.info(f"Registering plugin: {name}")
        self.plugins[name] = plugin

    def unregister(self, name: str):
        self.log.info(f"Unregistering plugin: {name}")
        del self.plugins[name]
