# Plugin registry for the Deep Learning Description Language
# Maps plugin names to plugin classes or factories.

class PluginRegistry:
    def __init__(self):
        self._plugins = {}  # name -> plugin class or factory

    def register_plugin(self, name, plugin):
        """
        Register a plugin.
        """
        self._plugins[name] = plugin

    def unregister_plugin(self, name):
        """
        Unregister a plugin by name.
        """
        if name in self._plugins:
            del self._plugins[name]

    def get_plugin(self, name):
        """
        Get a plugin by name.
        """
        return self._plugins.get(name)

    def list_plugins(self):
        """
        List all registered plugin names.
        """
        return list(self._plugins.keys())