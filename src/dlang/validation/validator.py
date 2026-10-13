# Validator for the Deep Learning Description Language AST
# Performs semantic checks on the AST.

class Validator:
    def __init__(self):
        pass

    def validate(self, ast):
        """
        Validate the AST and return a list of errors (empty if valid).
        """
        errors = []
        # TODO: Implement validation checks
        # For example:
        # - Check for undefined layers or blocks
        # - Check for duplicate names in the same scope
        # - Check compatibility between consecutive layers (output shape vs input shape)
        # - Check that required components (optimizer, loss) are present for training
        return errors