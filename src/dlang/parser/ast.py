# Abstract Syntax Tree nodes for the Deep Learning Description Language

class Node:
    def __init__(self):
        pass

class Model(Node):
    def __init__(self, name, inputs=None, outputs=None, layers=None, imports=None, blocks=None):
        self.name = name
        self.inputs = inputs or []  # List of Input nodes
        self.outputs = outputs or []  # List of Output nodes
        self.layers = layers or []  # List of layer statements
        self.imports = imports or []  # List of imported files
        self.blocks = blocks or []  # List of top-level Block nodes

class Input(Node):
    def __init__(self, shape, name):
        self.shape = shape  # Tuple of dimensions, e.g., (None, None, 1)
        self.name = name  # Variable name for the input tensor

class Output(Node):
    def __init__(self, source):
        self.source = source  # Variable name of the output tensor

class Layer(Node):
    def __init__(self, name, type, inputs, parameters=None):
        self.name = name  # Variable name for this layer
        self.type = type  # Layer type (e.g., 'Conv2D', 'Dense')
        self.inputs = inputs or []  # List of input tensors/variables
        self.parameters = parameters or {}  # Layer parameters (filters, kernel_size, etc.)

class Block(Node):
    def __init__(self, name, inputs=None, outputs=None, layers=None, parameters=None):
        self.name = name  # Block name
        self.inputs = inputs or []  # Input parameter names
        self.outputs = outputs or []  # Output parameter names
        self.layers = layers or []  # Layers within the block
        self.parameters = parameters or {}  # Block parameters

class Import(Node):
    def __init__(self, filepath):
        self.filepath = filepath  # Path to imported .dl file

class Repeat(Node):
    def __init__(self, count, body):
        self.count = count  # Number of repetitions
        self.body = body  # List of statements to repeat

class CustomLayer(Node):
    def __init__(self, name, inputs, equation_file, parameters=None):
        self.name = name  # Variable name for this custom layer
        self.inputs = inputs or []  # Input tensors
        self.equation_file = equation_file  # Path to equation file
        self.parameters = parameters or {}  # Parameters for the custom layer

# Additional nodes for expressions and operations
class Assignment(Node):
    def __init__(self, target, expression):
        self.target = target  # Target variable name
        self.expression = expression  # Expression to assign

class Variable(Node):
    def __init__(self, name):
        self.name = name  # Variable name

class FunctionCall(Node):
    def __init__(self, name, args=None, kwargs=None):
        self.name = name  # Function name
        self.args = args or []  # Positional arguments
        self.kwargs = kwargs or {}  # Keyword arguments

class Number(Node):
    def __init__(self, value):
        self.value = value  # Numeric value

class String(Node):
    def __init__(self, value):
        self.value = value  # String value

class Boolean(Node):
    def __init__(self, value):
        self.value = value  # Boolean value

class BinaryOperation(Node):
    def __init__(self, op, left, right):
        self.op = op  # Operator as string, e.g., 'PLUS', 'MINUS'
        self.left = left  # Left expression
        self.right = right  # Right expression