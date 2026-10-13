# Command to validate a .dl file for syntax and semantics
from dlang.parser.lexer import lex
from dlang.parser.parser import parse

def validate_file(args):
    input_file = args.input_file

    # Read the .dl file
    with open(input_file, 'r') as f:
        text = f.read()

    try:
        # Lex and parse
        tokens = lex(text)
        ast = parse(tokens)
        if ast is None:
            raise ValueError("Parser returned None")
        # TODO: Add semantic validation (e.g., check for undefined layers, shape compatibility)
        print(f"SUCCESS: {input_file} is syntactically valid.")
    except Exception as e:
        print(f"FAILURE: {input_file} is invalid.")
        print(f"Error: {e}")