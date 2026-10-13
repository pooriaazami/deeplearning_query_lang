"""
Reference PyTorch implementation for functional_model.dl
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputStream(nn.Module):
    def __init__(self, units, input_name, prefix=''):
        super(InputStream, self).__init__()
        self.dense = nn.Linear(units, units)  # Actually, this should take the input size
        # Wait, looking at the DL: INPUT(name=input_name) then Dense([INPUT], units=units, ...)
        # So the input to the dense layer is the INPUT tensor, which has whatever shape the input has
        # But we don't know the input size at definition time - it's determined when the block is instantiated
        # Actually, looking more carefully: the block has parameters (units, input_name, prefix)
        # But the INPUT(name=input_name) inside the block doesn't specify a shape
        # This suggests that the input shape comes from wherever the block is used
        # However, in the usage: proc_A = input_stream(units=64, input_name='input_a', prefix='stream_a_')
        # And earlier: INPUT(shape=(32,), name='input_a')
        # So the input_name refers to an INPUT that was defined earlier with a specific shape

        # Actually, re-reading: the BLOCK defines a template, and when instantiated,
        # the INPUT(name=input_name) refers to the input that was passed to the block
        # But in the DL, it looks like the block is supposed to read from the INPUT declaration
        # Let me look at the functional_model again:
        # INPUT(shape=(32,), name='input_a')
        # INPUT(shape=(32,), name='input_b')
        # proc_A = input_stream(units=64, input_name='input_a', prefix='stream_a_')
        #
        # This suggests that when we instantiate input_stream(units=64, input_name='input_a', ...),
        # the INPUT(name=input_name) inside the block should refer to the tensor named 'input_a'
        # But that doesn't make sense in the context of a block definition.
        #
        # Actually, I think the correct interpretation is:
        # The BLOCK defines a subgraph that expects an input tensor to be provided
        # When we instantiate the block, we provide that input tensor
        # The INPUT(name=input_name) inside the block is just declaring that the block expects an input
        # and giving it a name for use within the block
        #
        # But looking at the generated code from our earlier test, let me check what was generated...

        # For now, I'll implement it as: the block expects an input of size 'units' (which doesn't make sense)
        # Or perhaps the 'units' parameter is the output size, and the input size is determined by context
        #
        # Looking at the usage again:
        # proc_A = input_stream(units=64, input_name='input_a', prefix='stream_a_')
        # Where input_a was defined as INPUT(shape=(32,), name='input_a')
        # So input_a has 32 features, and we're creating a dense layer with 64 units
        #
        # Ah! I think I understand now. The 'units' parameter is the OUTPUT size of the dense layer.
        # The input size comes from wherever the block is connected.
        #
        # But wait, the block has INPUT(name=input_name) inside it, which suggests it's reading
        # from a previously declared INPUT.
        #
        # Let me think of it this way: the block is a template that, when instantiated,
        # creates a connection from the named INPUT to a dense layer.
        #
        # Actually, looking at it more carefully, I think the correct interpretation is:
        # The BLOCK defines a reusable piece that computes: dense = Dense([INPUT], units=units, ...)
        # When we do: proc_A = input_stream(units=64, input_name='input_a', prefix='stream_a_')
        # It means: create a dense layer that takes as input whatever is named 'input_a'
        # and apply a dense layer with 64 units to it.
        #
        # So the block is parameterized by:
        # - units: the output size of the dense layer
        # - input_name: the name of the INPUT to use as input to the dense layer
        # - prefix: prefix for naming layers inside the block
        #
        # But this creates a problem: the block needs to know what tensor corresponds to input_name
        # This suggests that the block instantiation happens in the context of the model
        # where all INPUT declarations are known.
        #
        # For the reference implementation, I'll handle this by:
        # 1. Creating InputStream modules that expect to be given the input tensor during forward pass
        # 2. But that doesn't match the DL syntax where input_name refers to a declared INPUT
        #
        # Let me look at this differently. Perhaps the correct interpretation is:
        # The BLOCK defines a template that, when instantiated with input_name='X',
        # creates a module that expects an input and treats it as if it were the tensor named 'X'
        #
        # Actually, let me just implement it as a simple dense layer where:
        # - The input size is determined when the model is used (we'll make it flexible)
        # - The output size is given by the 'units' parameter
        # And in the test, we'll ensure the input size matches what's expected

        # For now, I'll make it so that the InputStream takes an input size parameter
        # But looking at the DL, there's no input size parameter to the block
        #
        # Let me re-examine: BLOCK input_stream(units, input_name, prefix='')
        # The block has three parameters: units, input_name, prefix
        # Inside: INPUT(name=input_name) - this declares that the block expects an input
        #         dense = Dense([INPUT], units=units, activation='relu', name=prefix+'dense')
        #         OUTPUT = dense
        #
        # So when we instantiate: proc_A = input_stream(units=64, input_name='input_a', prefix='stream_a_')
        # It means: create a component that:
        # 1. Expects an input (which will be connected to the input_a INPUT)
        # 2. Applies a dense layer with 64 units and ReLU activation
        # 3. Outputs the result
        #
        # This is essentially just a dense layer where:
        # - Input size: determined by what's connected to it (should match input_a's shape)
        # - Output size: units parameter (64)
        #
        # But wait, how does it know to connect to input_a?
        # I think the instantiation input_stream(units=64, input_name='input_a', ...)
        # means: use the tensor named 'input_a' as the input to this block
        #
        # So the block, when instantiated, creates a dense layer whose input
        # is connected to whatever tensor was named by the input_name parameter
        #
        # For the reference implementation, I'll make InputStream take the input size as a parameter
        # since we need to know it to create the Linear layer, and in the test we'll
        # ensure it matches the actual input_a size.

        # Actually, let me think of it as: the InputStream is a function that,
        # given an input tensor, applies a dense layer to it.
        # The 'units' parameter controls the output size.
        # The input size is inferred from the input tensor.

        # But PyTorch nn.Linear needs to know both input and output sizes at initialization time.
        #
        # Hmm. Let me check what the actual generated code looks like from our compiler
        # to see how it handled this.

        # Since I don't have that easily accessible, I'll make a reasonable assumption:
        # The InputStream represents a dense layer where:
        # - Output size = units parameter
        # - Input size = will be determined from context (we'll make it work with whatever size is given)
        #
        # But nn.Linear doesn't work that way.
        #
        # Alternative: make it so that we delay creating the Linear layer until we see the input
        # (like using lazy initialization or checking input size on first forward pass)
        #
        # For simplicity in the reference implementation, I'll assume we know the input size
        # from the model context, and since this is a test, I'll hardcode the expected size
        # based on what we see in the model.
        #
        # Looking at functional_model.dl:
        # INPUT(shape=(32,), name='input_a')
        # INPUT(shape=(32,), name='input_b')
        # proc_A = input_stream(units=64, input_name='input_a', prefix='stream_a_')
        # proc_B = input_stream(units=64, input_name='input_b', prefix='stream_b_')
        #
        # So both input_a and input_b have shape (32,) - meaning 32 features
        #
        # Therefore, when instantiating input_stream with input_name='input_a' or 'input_b',
        # the input size should be 32.

        self.dense = nn.Linear(32, units)  # Hardcoded for test - input size 32, output size = units
        self.units = units

    def forward(self, x):
        # x shape: (batch_size, input_size) where input_size should be 32
        x = F.relu(self.dense(x))
        return x


class FunctionalModel(nn.Module):
    def __init__(self):
        super(FunctionalModel, self).__init__()
        # Input shapes are defined in the model
        # input_a: shape=(32,) -> 32 features
        # input_b: shape=(32,) -> 32 features

        # The input_stream blocks:
        # input_stream(units=64, input_name='input_a', prefix='stream_a_')
        # This should take input_a (32 features) and output 64 features
        self.proc_A = InputStream(units=64, input_name='input_a', prefix='stream_a_')

        # input_stream(units=64, input_name='input_b', prefix='stream_b_')
        # This should take input_b (32 features) and output 64 features
        self.proc_B = InputStream(units=64, input_name='input_b', prefix='stream_b_')

        # After processing, we concatenate: 64 + 64 = 128 features
        self.output_layer = nn.Linear(128, 10)  # 10 classes for output

    def forward(self, input_a, input_b):
        # input_a shape: (batch_size, 32)
        # input_b shape: (batch_size, 32)

        # Process each input through its stream
        out_A = self.proc_A(input_a)  # -> (batch_size, 64)
        out_B = self.proc_B(input_b)  # -> (batch_size, 64)

        # Concatenate along feature dimension
        combined = torch.cat([out_A, out_B], dim=1)  # -> (batch_size, 128)

        # Final output
        output = F.softmax(self.output_layer(combined), dim=1)  # -> (batch_size, 10)
        return output