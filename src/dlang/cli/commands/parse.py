# Command to parse a .dl file and generate code
import os
from .. import __version__  # for version info
from dlang.parser.lexer import lex
from dlang.parser.parser import parse
from dlang.codegen.generator import Generator

def parse_file(args):
    input_file = args.input_file
    output_dir = args.output or os.path.dirname(input_file) or '.'
    backend = args.backend

    # Read the .dl file
    with open(input_file, 'r') as f:
        text = f.read()

    # Lex and parse
    tokens = lex(text)
    ast = parse(tokens)

    # Generate code
    generator = Generator(backend=backend)
    code = generator.generate(ast)

    # Determine output file name
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_dir, f"{base_name}_{backend}.py")

    # Write the generated code
    with open(output_file, 'w') as f:
        f.write(code)

    print(f"Parsed {input_file} and generated {output_file}")