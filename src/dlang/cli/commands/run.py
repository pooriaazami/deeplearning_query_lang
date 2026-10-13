# Command to parse .dl file, generate code, and run it
import os
import subprocess
import sys
from dlang.parser.lexer import lex
from dlang.parser.parser import parse
from dlang.codegen.generator import Generator

def run_file(args):
    input_file = args.input_file
    backend = args.backend
    device = args.device  # not used yet, but could be passed to the generated code

    # Read the .dl file
    with open(input_file, 'r') as f:
        text = f.read()

    # Lex and parse
    tokens = lex(text)
    ast = parse(tokens)

    # Generate code
    generator = Generator(backend=backend)
    code = generator.generate(ast)

    # Determine output file name (temporary)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{base_name}_{backend}_generated.py"

    # Write the generated code
    with open(output_file, 'w') as f:
        f.write(code)

    # Run the generated code
    # We'll use the same Python interpreter
    result = subprocess.run([sys.executable, output_file], capture_output=True, text=True)

    # Print output
    if result.stdout:
        print("STDOUT:")
        print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)

    print(f"Generated code exited with return code {result.returncode}")

    # Optionally, remove the generated file? Or leave it for inspection.
    # We'll leave it for now.