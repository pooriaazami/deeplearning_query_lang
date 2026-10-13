# Plugin manager for the Deep Learning Description Language
# Handles discovery, loading, and management of plugins.

import pkg_resources
from .registry import PluginRegistry

class PluginManager:
    def __init__(self):
        self.registry = PluginRegistry()
        self._loaded = False

    def discover_plugins(self):
        """
        Discover plugins installed via entry points or in local directories.
        """
        # Discover via entry points (e.g., 'dlang.plugins')
        for entry_point in pkg_resources.iter_entry_points('dlang.plugins'):
            plugin = entry_point.load()
            self.registry.register_plugin(entry_point.name, plugin)

        # TODO: Also scan local directories for plugins

    def get_plugin(self, name):
        """
        Get a plugin by name.
        """
        if not self._loaded:
            self.discover_plugins()
            self._loaded = True
        return self.registry.get_plugin(name)

    def list_plugins(self):
        """
        List all available plugins.
        """
        if not self._loaded:
            self.discover_plugins()
            self._loaded = True
        return self.registry.list_plugins()