# Lexer for the Deep Learning Description Language
# Tokens the input text into meaningful symbols for parsing

import re
from typing import List, Tuple

# Token specifications
TOKEN_SPECS = [
    # Keywords (case-sensitive)
    ('MODEL', r'\bMODEL\b'),
    ('BLOCK', r'\bBLOCK\b'),
    ('INPUT', r'\bINPUT\b'),
    ('OUTPUT', r'\bOUTPUT\b'),
    ('END', r'\bEND\b'),
    ('REPEAT', r'\bREPEAT\b'),
    ('times', r'\btimes\b'),
    ('CUSTOM', r'\bCUSTOM\b'),
    ('IMPORT', r'\bIMPORT\b'),

    # Layer types and operations (proper case as per convention)
    ('CONV2D', r'\bConv2D\b'),
    ('MAXPOOL2D', r'\bMaxPool2D\b'),
    ('UPSAMPLING2D', r'\bUpSampling2D\b'),
    ('GLOBALAVERAGEPOOLING2D', r'\bGlobalAveragePooling2D\b'),
    ('DENSE', r'\bDense\b'),
    ('FLATTEN', r'\bFlatten\b'),
    ('RESHAPE', r'\bReshape\b'),
    ('BATCHNORMALIZATION', r'\bBatchNormalization\b'),
    ('ACTIVATION', r'\bActivation\b'),
    ('ADD', r'\bAdd\b'),
    ('CONCATENATE', r'\bConcatenate\b'),
    ('ELEMENTWISEMULTIPLY', r'\bElementwiseMultiply\b'),
    ('MULTIPLY', r'\bMultiply\b'),
    ('LSTM', r'\bLSTM\b'),
    ('GRU', r'\bGRU\b'),
    ('ZEROS', r'\bZeros\b'),
    ('RANDOMNORMAL', r'\bRandomNormal\b'),
    ('EXP', r'\bExp\b'),

    # Literals
    ('STRING', r'"[^"]*"|\'[^\']*\''),  # Both double and single quoted strings
    ('NUMBER', r'\b\d+(\.\d+)?\b'),  # Integers and floats
    ('BOOLEAN', r'\b(True|False)\b'),  # Boolean literals

    # Identifiers (variable names, function names, etc.)
    ('IDENTIFIER', r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),

    # Comments
    ('COMMENT', r'#.*'),  # Comments: from # to end of line

    # Delimiters and operators
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('LBRACKET', r'\['),
    ('RBRACKET', r'\]'),
    ('LBRACE', r'\{'),
    ('RBRACE', r'\}'),
    ('COMMA', r','),
    ('DOT', r'\.'),
    ('ASSIGN', r'='),
    ('COLON', r':'),
    ('PLUS', r'\+'),  # Addition operator
    ('MINUS', r'-'),  # Subtraction operator
    ('NEWLINE', r'\n'),
    ('SKIP', r'[ \t]+'),  # Skip spaces and tabs
    ('MISMATCH', r'.'),   # Any other character
]

# Compile regex patterns
TOKEN_REGEX = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPECS)
GET_TOKEN = re.compile(TOKEN_REGEX).match


def lex(text: str) -> List[Tuple[str, str]]:
    """
    Lex the input text into a list of tokens.

    Args:
        text: The input .dl file content

    Returns:
        List of (token_type, token_value) tuples
    """
    tokens = []
    pos = 0
    line_num = 1

    while pos < len(text):
        match = GET_TOKEN(text, pos)
        if not match:
            raise RuntimeError(f'Unexpected character at position {pos}: {text[pos]}')

        token_type = match.lastgroup
        token_value = match.group(token_type)

        if token_type == 'NEWLINE':
            tokens.append((token_type, token_value))
            line_num += 1
        elif token_type in ('SKIP', 'COMMENT'):
            pass  # Skip whitespace and comments
        elif token_type == 'MISMATCH':
            raise RuntimeError(f'Unexpected token {token_value} at line {line_num}')
        else:
            tokens.append((token_type, token_value))

        pos = match.end()

    return tokens