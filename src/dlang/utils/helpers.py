# Helper functions for the Deep Learning Description Language
# This module contains utility functions used across the project.

def sanitize_name(name):
    """
    Sanitize a name to be used as an identifier.
    """
    # Replace spaces and special characters with underscores
    import re
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)

def camel_to_snake(name):
    """
    Convert camelCase to snake_case.
    """
    import re
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

def snake_to_camel(name):
    """
    Convert snake_case to camelCase.
    """
    return ''.join(word.capitalize() for word in name.split('_'))