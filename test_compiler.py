#!/usr/bin/env python3
"""
Test script to verify the DL compiler implementation.
"""

import sys
import os

# Add the src directory to the path so we can import dlang modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dlang.parser.lexer import lex
from dlang.parser.parser import Parser
from dlang.codegen.generator import Generator

def test_simple_model():
    """Test compiling the simple_model.dl file"""
    print("Testing simple_model.dl...")

    # Read the file
    with open('test/basic/simple_model.dl', 'r') as f:
        content = f.read()

    print("File content:")
    print(content)
    print()

    # Lex the content
    print("Lexing...")
    tokens = lex(content)
    print(f"Tokens: {tokens}")
    print()

    # Parse the tokens
    print("Parsing...")
    parser = Parser(tokens)
    ast = parser.parse()
    print(f"AST: {ast}")
    print(f"Model name: {ast.name}")
    print(f"Number of inputs: {len(ast.inputs)}")
    print(f"Number of outputs: {len(ast.outputs)}")
    print(f"Number of layers: {len(ast.layers)}")
    print()

    # Generate code
    print("Generating PyTorch code...")
    generator = Generator(backend='pytorch')
    pytorch_code = generator.generate(ast)
    print("Generated PyTorch code:")
    print(pytorch_code)
    print()

    # Check if the code is valid Python by trying to compile it
    try:
        compile(pytorch_code, '<generated>', 'exec')
        print("[OK] Generated code compiles successfully")
    except SyntaxError as e:
        print(f"[FAIL] Generated code has syntax error: {e}")

    return pytorch_code

def test_loop_model():
    """Test compiling the loop_model.dl file"""
    print("\n" + "="*50)
    print("Testing loop_model.dl...")

    # Read the file
    with open('test/basic/loop_model.dl', 'r') as f:
        content = f.read()

    print("File content:")
    print(content)
    print()

    # Lex the content
    print("Lexing...")
    tokens = lex(content)
    print(f"Number of tokens: {len(tokens)}")
    print()

    # Parse the tokens
    print("Parsing...")
    parser = Parser(tokens)
    ast = parser.parse()
    print(f"AST: {ast}")
    print(f"Model name: {ast.name}")
    print(f"Number of inputs: {len(ast.inputs)}")
    print(f"Number of outputs: {len(ast.outputs)}")
    print(f"Number of layers: {len(ast.layers)}")
    print()

    # Generate code
    print("Generating PyTorch code...")
    generator = Generator(backend='pytorch')
    pytorch_code = generator.generate(ast)
    print("Generated PyTorch code:")
    print(pytorch_code)
    print()

    # Check if the code is valid Python by trying to compile it
    try:
        compile(pytorch_code, '<generated>', 'exec')
        print("[OK] Generated code compiles successfully")
    except SyntaxError as e:
        print(f"[FAIL] Generated code has syntax error: {e}")

    return pytorch_code

if __name__ == '__main__':
    try:
        test_simple_model()
        test_loop_model()
        print("\n" + "="*50)
        print("All tests completed!")
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()