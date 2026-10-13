# Abstract Syntax Tree nodes for the Deep Learning Description Language

class Node:
    pass

class Model(Node):
    def __init__(self, name, layers=None, inputs=None, outputs=None):
        self.name = name
        self.layers = layers or []
        self.inputs = inputs or []
        self.outputs = outputs or []

class Layer(Node):
    def __init__(self, name, type, parameters=None):
        self.name = name
        self.type = type
        self.parameters = parameters or {}

class Block(Node):
    def __init__(self, name, layers=None, parameters=None):
        self.name = name
        self.layers = layers or []
        self.parameters = parameters or {}

class Dataset(Node):
    def __init__(self, name, source, preprocessing=None, split=None):
        self.name = name
        self.source = source
        self.preprocessing = preprocessing or []
        self.split = split or {}

class PreprocessingStep(Node):
    def __init__(self, type, parameters=None):
        self.type = type
        self.parameters = parameters or {}

class Split(Node):
    def __init__(self, type, ratio=None, seed=None):
        self.type = type  # 'random', 'stratified', 'timeseries'
        self.ratio = ratio
        self.seed = seed

class TrainingConfig(Node):
    def __init__(self, optimizer=None, loss=None, metrics=None, epochs=None, batch_size=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or []
        self.epochs = epochs
        self.batch_size = batch_size

class Optimizer(Node):
    def __init__(self, type, parameters=None):
        self.type = type
        self.parameters = parameters or {}

class Loss(Node):
    def __init__(self, type, parameters=None):
        self.type = type
        self.parameters = parameters or {}

class Metric(Node):
    def __init__(self, type, parameters=None):
        self.type = type
        self.parameters = parameters or {}

class Logger(Node):
    def __init__(self, type, parameters=None):
        self.type = type
        self.parameters = parameters or {}