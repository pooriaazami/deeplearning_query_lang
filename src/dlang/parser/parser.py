# Parser for the Deep Learning Description Language
# Parses tokenized input and produces an AST

from typing import List, Optional, Tuple, Any
from .ast import *
from .lexer import lex

class Parser:
    def __init__(self, tokens: List[Tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[0] if self.tokens else None

    def advance(self):
        """Move to the next token"""
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
        else:
            self.current_token = None

    def peek(self, offset: int = 1) -> Optional[Tuple[str, str]]:
        """Peek at a token ahead without consuming it"""
        peek_pos = self.pos + offset
        if peek_pos < len(self.tokens):
            return self.tokens[peek_pos]
        return None

    def expect(self, token_type: str) -> Tuple[str, str]:
        """Consume and return the current token if it matches the expected type"""
        if self.current_token and self.current_token[0] == token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            expected = token_type
            actual = self.current_token[0] if self.current_token else None
            raise SyntaxError(f'Expected {expected}, got {actual} at position {self.pos}')

    def parse(self) -> Model:
        """Parse the entire token stream and return a Model AST node"""
        # Skip any initial newlines or whitespace
        while self.current_token and self.current_token[0] == 'NEWLINE':
            self.advance()

        # Parse imports first (if any)
        imports = []
        while self.current_token and self.current_token[0] == 'IMPORT':
            imports.append(self.parse_import())

        # Parse top-level blocks (if any)
        blocks = []
        while self.current_token and self.current_token[0] == 'BLOCK':
            blocks.append(self.parse_block())

        # Skip any newlines or whitespace before model
        while self.current_token and self.current_token[0] == 'NEWLINE':
            self.advance()

        # Parse the model
        model = self.parse_model()

        # Attach imports and blocks to the model
        model.imports = imports
        model.blocks = blocks

        return model

    def parse_import(self) -> Import:
        """Parse an import statement: IMPORT "filepath" """
        self.expect('IMPORT')
        string_token = self.expect('STRING')
        filepath = string_token[1].strip('"')  # Remove quotes
        self.expect('NEWLINE')  # Expect newline after import
        return Import(filepath)

    def parse_model(self) -> Model:
        """Parse a model definition: MODEL name ... END"""
        self.expect('MODEL')
        name_token = self.expect('IDENTIFIER')
        model_name = name_token[1]

        self.expect('NEWLINE')

        # Parse model body
        inputs = []
        outputs = []
        layers = []

        while self.current_token and self.current_token[0] != 'END':
            if self.current_token[0] == 'INPUT':
                inputs.append(self.parse_input())
            elif self.current_token[0] == 'OUTPUT':
                outputs.append(self.parse_output())
            elif self.current_token[0] == 'BLOCK':
                # Parse block definition
                block = self.parse_block()
                layers.append(block)
            elif self.current_token[0] == 'REPEAT':
                # Parse repeat loop
                repeat_stmt = self.parse_repeat()
                layers.append(repeat_stmt)
            else:
                # Parse layer statement or assignment
                layer_or_assign = self.parse_statement()
                if layer_or_assign:
                    layers.append(layer_or_assign)

            # Skip newlines between statements
            while self.current_token and self.current_token[0] == 'NEWLINE':
                self.advance()

        self.expect('END')
        # Optionally consume newline after END if present
        if self.current_token and self.current_token[0] == 'NEWLINE':
            self.advance()

        return Model(
            name=model_name,
            inputs=inputs,
            outputs=outputs,
            layers=layers
        )

    def parse_input(self) -> Input:
        """Parse an input declaration: INPUT(shape=(...), name='...')"""
        self.expect('INPUT')
        self.expect('LPAREN')

        # Parse shape parameter
        self.expect('IDENTIFIER')  # 'shape'
        self.expect('ASSIGN')
        self.expect('LPAREN')  # Start of shape tuple

        shape_elements = []
        while self.current_token and self.current_token[0] != 'RPAREN':
            if self.current_token[0] == 'NUMBER':
                num_token = self.expect('NUMBER')
                shape_elements.append(int(num_token[1]) if '.' not in num_token[1] else float(num_token[1]))
                # After processing a number, look ahead to see if comma is shape separator or parameter separator
                if self.current_token and self.current_token[0] == 'COMMA':
                    self.expect('COMMA')  # Consume the comma
                    # Look ahead to see what comes next
                    if self.current_token and (self.current_token[0] == 'NUMBER' or
                                              (self.current_token[0] == 'IDENTIFIER' and self.current_token[1] == 'None') or
                                              self.current_token[0] == 'RPAREN'):
                        # Next token is a valid shape element or closing parenthesis - continue parsing shape elements
                        continue
                    else:
                        # Next token is neither a shape element nor closing parenthesis - this comma is the parameter separator
                        break
            elif self.current_token[0] == 'IDENTIFIER' and self.current_token[1] == 'None':
                self.expect('IDENTIFIER')  # 'None'
                shape_elements.append(None)
                # After processing None, look ahead to see if comma is shape separator or parameter separator
                if self.current_token and self.current_token[0] == 'COMMA':
                    self.expect('COMMA')  # Consume the comma
                    # Look ahead to see what comes next
                    if self.current_token and (self.current_token[0] == 'NUMBER' or
                                              (self.current_token[0] == 'IDENTIFIER' and self.current_token[1] == 'None') or
                                              self.current_token[0] == 'RPAREN'):
                        # Next token is a valid shape element or closing parenthesis - continue parsing shape elements
                        continue
                    else:
                        # Next token is neither a shape element nor closing parenthesis - this comma is the parameter separator
                        break
            elif self.current_token[0] == 'COMMA':
                # If we see a comma and we haven't processed any elements yet,
                # this is unexpected
                if not shape_elements:
                    raise SyntaxError(f'Unexpected token in shape: {self.current_token}')
                # If we have processed elements, treat this comma as the parameter separator
                break
            else:
                # Unexpected token in shape
                raise SyntaxError(f'Unexpected token in shape: {self.current_token}')

        # Only expect RPAREN if we didn't break out early due to a comma that was actually a parameter separator
        # If we broke out after seeing a comma that was a parameter separator, we've already consumed it
        # If we exited the loop normally by seeing RPAREN, we need to consume it
        if self.current_token and self.current_token[0] == 'RPAREN':
            self.expect('RPAREN')  # End of shape tuple
            # After consuming the shape tuple's closing parenthesis, we need to consume the comma
            # that separates the shape parameter from the name parameter
            self.expect('COMMA')  # Comma after shape
        # Otherwise, we've already consumed the comma that separates shape from name parameters

        # Parse name parameter
        self.expect('IDENTIFIER')  # 'name'
        self.expect('ASSIGN')
        if self.current_token[0] == 'STRING':
            name_token = self.expect('STRING')
            name = name_token[1].strip('"')
        elif self.current_token[0] == 'IDENTIFIER':
            # Treat the identifier as a string by using its identifier name
            id_token = self.expect('IDENTIFIER')
            name = id_token[1]
        else:
            raise SyntaxError(f'Expected STRING or IDENTIFIER for name, got {self.current_token}')

        self.expect('RPAREN')  # End of INPUT call
        self.expect('NEWLINE')  # Expect newline after input

        return Input(shape=tuple(shape_elements), name=name)

    def parse_output(self) -> Output:
        """Parse an output declaration: OUTPUT = tensor_name"""
        self.expect('OUTPUT')
        self.expect('ASSIGN')
        # Parse the expression (could be variable reference or function call)
        expr = self.parse_expression()
        var_name = None

        if isinstance(expr, Variable):
            # Simple variable reference: OUTPUT = var_name
            var_name = expr.name
        elif isinstance(expr, FunctionCall):
            # Function call: OUTPUT = function_call(...)
            # Look for 'name' keyword argument to determine the variable name
            name_arg = None
            # Check for common name parameter names
            for param_name in ['name', 'Name', 'NAME']:
                if param_name in expr.kwargs:
                    name_arg = expr.kwargs[param_name]
                    break

            # If still not found, look for any string-valued keyword argument
            if name_arg is None:
                for param_name, arg_value in expr.kwargs.items():
                    if isinstance(arg_value, String):
                        name_arg = arg_value
                        break

            if name_arg is not None:
                if isinstance(name_arg, String):
                    var_name = name_arg.value
                else:
                    raise SyntaxError(f'OUTPUT function call name parameter must be a string literal, got {type(name_arg).__name__}')
            else:
                raise SyntaxError(f'OUTPUT function call must include a name parameter')
        else:
            raise SyntaxError(f'OUTPUT statement expects a variable name or function call, got {type(expr).__name__}')

        self.expect('NEWLINE')
        return Output(source=var_name)

    def parse_block(self) -> Block:
        """Parse a block definition: BLOCK name(...) ... END"""
        self.expect('BLOCK')
        name_token = self.expect('IDENTIFIER')
        block_name = name_token[1]

        self.expect('LPAREN')

        # Parse block parameters
        params = []
        while self.current_token and self.current_token[0] != 'RPAREN':
            if self.current_token[0] == 'IDENTIFIER':
                param_token = self.expect('IDENTIFIER')
                params.append(param_token[1])
                if self.current_token and self.current_token[0] == 'COMMA':
                    self.expect('COMMA')
            else:
                # Skip unexpected tokens
                self.advance()

        self.expect('RPAREN')
        # Actually, looking at samples, it's BLOCK name(...) then newline, not =
        self.expect('NEWLINE')

        # Parse block body
        inputs = []  # Will store input parameter names
        outputs = []  # Will store output variable names
        layers = []

        # First, we need to parse the parameter declarations (they come as INPUT statements)
        # But looking at the samples, block parameters are referenced by name in INPUT statements inside the block
        # For now, we'll collect all INPUT statements as inputs to the block

        while self.current_token and self.current_token[0] != 'END':
            if self.current_token[0] == 'INPUT':
                inp = self.parse_input()
                inputs.append(inp.name)  # Store the input variable name
                layers.append(('input', inp))  # Store as special input layer for now
            elif self.current_token[0] == 'OUTPUT':
                out = self.parse_output()
                outputs.append(out.source)  # Store the output variable name
                layers.append(('output', out))  # Store as special output layer
            elif self.current_token[0] == 'BLOCK':
                nested_block = self.parse_block()
                layers.append(('block', nested_block))
            else:
                layer_or_assign = self.parse_statement()
                if layer_or_assign:
                    layers.append(layer_or_assign)

            # Skip newlines
            while self.current_token and self.current_token[0] == 'NEWLINE':
                self.advance()

        self.expect('END')
        self.expect('NEWLINE')

        return Block(
            name=block_name,
            inputs=inputs,
            outputs=outputs,
            layers=layers
        )

    def parse_repeat(self) -> Repeat:
        """Parse a repeat loop: REPEAT n times: ... END"""
        self.expect('REPEAT')
        count_token = self.expect('NUMBER')
        count = int(count_token[1])
        self.expect('times')  # 'times'
        self.expect('COLON')
        self.expect('NEWLINE')

        # Parse body
        body = []
        while self.current_token and self.current_token[0] != 'END':
            if self.current_token[0] == 'INPUT':
                body.append(self.parse_input())
            elif self.current_token[0] == 'OUTPUT':
                body.append(self.parse_output())
            elif self.current_token[0] == 'BLOCK':
                body.append(self.parse_block())
            else:
                stmt = self.parse_statement()
                if stmt:
                    body.append(stmt)

            # Skip newlines
            while self.current_token and self.current_token[0] == 'NEWLINE':
                self.advance()

        self.expect('END')
        self.expect('NEWLINE')

        return Repeat(count=count, body=body)

    def parse_statement(self):
        """Parse a statement which could be a layer instantiation, assignment, or function call"""
        # Look ahead to see if this is an assignment (contains =) or just a layer call/function call
        # Actually, looking at samples, all layer instantiations are assignments: layer = LayerType(...)
        # But some functions like INPUT, OUTPUT can be called standalone

        # Parse left-hand side (identifier)
        if self.current_token and self.current_token[0] == 'IDENTIFIER':
            # Save current position to backtrack if needed
            # Actually, let's look ahead to see what comes after the identifier
            first_token = self.current_token

            # Check if this is an assignment: IDENTIFIER ASSIGN ...
            if self.peek() and self.peek()[0] == 'ASSIGN':
                # This is an assignment: var_name = expression
                var_token = self.expect('IDENTIFIER')
                var_name = var_token[1]
                self.expect('ASSIGN')  # Consume the =

                # Parse the right-hand side (layer instantiation or expression)
                expr = self.parse_expression()

                self.expect('NEWLINE')

                return Assignment(target=var_name, expression=expr)
            elif self.peek() and self.peek()[0] == 'LPAREN':
                # This is a function call statement: function_name(...)
                # Parse it as a function call and expect it to be followed by NEWLINE
                func_call = self.parse_function_call(first_token[1])
                self.expect('NEWLINE')
                # For standalone function calls like INPUT, we don't need to return anything meaningful
                # but we'll return the function call for consistency
                return func_call
            else:
                # This is just a variable reference
                var_token = self.expect('IDENTIFIER')
                var_name = var_token[1]
                self.expect('NEWLINE')
                return Variable(name=var_name)
        else:
            # Not a valid statement, skip and advance
            if self.current_token:
                self.advance()
            return None

    def parse_expression(self):
        """Parse an expression (layer instantiation, function call, or variable)"""
        return self.parse_additive()

    def parse_additive(self):
        """Parse additive expressions: handles PLUS and MINUS"""
        left = self.parse_multiplicative()
        while self.current_token and self.current_token[0] in ('PLUS', 'MINUS'):
            op = self.current_token[0]
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOperation(op=op, left=left, right=right)
        return left

    def parse_multiplicative(self):
        """Parse multiplicative expressions: handles MULTIPLY"""
        left = self.parse_power()
        while self.current_token and self.current_token[0] == 'MULTIPLY':
            op = self.current_token[0]
            self.advance()
            right = self.parse_power()
            left = BinaryOperation(op=op, left=left, right=right)
        return left

    def parse_power(self):
        """Parse power expressions: handles exponentiation (right associative)"""
        left = self.parse_unary()
        # Note: We don't have a POWER token in the lexer, so we just return the unary expression.
        # If we had a POWER token, we would handle it here (right associative).
        return left

    def parse_unary(self):
        """Parse unary expressions: handles NOT and unary MINUS"""
        # Handle unary operators: NOT, MINUS
        if self.current_token and self.current_token[0] in ('NOT', 'MINUS'):
            op = self.current_token[0]
            self.advance()
            operand = self.parse_unary()
            # For now, we don't have a UnaryOperation node in the AST.
            # Handle common cases: negate numeric literals
            if op == 'MINUS' and isinstance(operand, Number):
                return Number(value=-operand.value)
            # For boolean literals, NOT flips the value
            elif op == 'NOT' and isinstance(operand, Boolean):
                return Boolean(value=not operand.value)
            # For other cases, return the operand as-is (temporary workaround)
            return operand
        return self.parse_primary()

    def parse_primary(self):
        """Parse primary expressions: literals, lists, special tokens, function calls, variables, and parentheses"""
        if not self.current_token:
            raise SyntaxError(f'Expected identifier, got {self.current_token}')

        # Handle special tokens that can be expressions (INPUT, OUTPUT)
        if self.current_token[0] in ('INPUT', 'OUTPUT'):
            special_token = self.current_token
            self.advance()
            # Return a variable node representing the special token
            return Variable(name=special_token[0])

        # Handle literals
        if self.current_token[0] in ('STRING', 'NUMBER', 'BOOLEAN'):
            return self.parse_literal()

        # Handle list expression: [expr1, expr2, ...]
        if self.current_token[0] == 'LBRACKET':
            self.expect('LBRACKET')
            exprs = []
            while self.current_token and self.current_token[0] != 'RBRACKET':
                exprs.append(self.parse_expression())
                if self.current_token and self.current_token[0] == 'COMMA':
                    self.expect('COMMA')
            self.expect('RBRACKET')
            # For now, we assume a list with one element and return that element.
            if len(exprs) == 1:
                return exprs[0]
            else:
                # If there are multiple elements, we return the first one (hack)
                return exprs[0] if exprs else None

        # Handle parentheses for grouping
        if self.current_token[0] == 'LPAREN':
            self.expect('LPAREN')
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr

        # Look for layer type or function call
        known_function_types = {
            'CONV2D', 'MAXPOOL2D', 'UPSAMPLING2D', 'GLOBALAVERAGEPOOLING2D',
            'DENSE', 'FLATTEN', 'RESHAPE', 'BATCHNORMALIZATION', 'ACTIVATION',
            'ADD', 'CONCATENATE', 'ELEMENTWISEMULTIPLY', 'MULTIPLY',
            'LSTM', 'GRU', 'ZEROS', 'RANDOMNORMAL', 'EXP',
            'CUSTOM'
        }

        if self.current_token[0] in known_function_types:
            # This is a known layer type token (like DENSE, Conv2D, etc.)
            func_name = self.current_token[1]
            return self.parse_function_call(func_name)
        elif self.current_token[0] == 'IDENTIFIER':
            # Look ahead to see if it's a function call (identifier followed by '(')
            if self.peek() and self.peek()[0] == 'LPAREN':
                # Treat it as a function call
                func_name = self.current_token[1]
                return self.parse_function_call(func_name)
            else:
                # It's a variable reference
                var_token = self.expect('IDENTIFIER')
                return Variable(name=var_token[1])
        else:
            raise SyntaxError(f'Expected identifier, got {self.current_token}')

    def parse_function_call(self, name: str):
        """Parse a function call: name(args, kwargs)"""
        # Consume the function name token (can be IDENTIFIER or specific function type)
        if self.current_token:
            if self.current_token[0] == 'IDENTIFIER':
                # Function name is an identifier (e.g., "DENSE" all caps)
                actual_name = self.current_token[1]
                if actual_name.lower() != name.lower():
                    raise SyntaxError(f'Expected function name {name}, got {actual_name}')
                self.expect('IDENTIFIER')
            elif self.current_token[0] in {
                'CONV2D', 'MAXPOOL2D', 'UPSAMPLING2D', 'GLOBALAVERAGEPOOLING2D',
                'DENSE', 'FLATTEN', 'RESHAPE', 'BATCHNORMALIZATION', 'ACTIVATION',
                'ADD', 'CONCATENATE', 'ELEMENTWISEMULTIPLY', 'MULTIPLY',
                'LSTM', 'GRU', 'ZEROS', 'RANDOMNORMAL', 'EXP',
                'CUSTOM'
            }:
                # Function name is a specific function type token (e.g., "Dense")
                actual_name = self.current_token[1]
                expected_name_map = {
                    'CONV2D': 'Conv2D',
                    'MAXPOOL2D': 'MaxPool2D',
                    'UPSAMPLING2D': 'UpSampling2D',
                    'GLOBALAVERAGEPOOLING2D': 'GlobalAveragePooling2D',
                    'DENSE': 'Dense',
                    'FLATTEN': 'Flatten',
                    'RESHAPE': 'Reshape',
                    'BATCHNORMALIZATION': 'BatchNormalization',
                    'ACTIVATION': 'Activation',
                    'ADD': 'Add',
                    'CONCATENATE': 'Concatenate',
                    'ELEMENTWISEMULTIPLY': 'ElementwiseMultiply',
                    'MULTIPLY': 'Multiply',
                    'LSTM': 'LSTM',
                    'GRU': 'GRU',
                    'ZEROS': 'Zeros',
                    'RANDOMNORMAL': 'RandomNormal',
                    'EXP': 'Exp',
                    'CUSTOM': 'CUSTOM'
                }
                expected_name = expected_name_map.get(self.current_token[0])
                if expected_name and expected_name.lower() != name.lower():
                    raise SyntaxError(f'Expected function name {name}, got {actual_name}')
                self.advance()
            else:
                raise SyntaxError(f'Expected identifier or function type, got {self.current_token}')
        else:
            raise SyntaxError(f'Expected function name {name}, got nothing')

        self.expect('LPAREN')

        args = []
        kwargs = {}

        # Parse arguments
        while self.current_token and self.current_token[0] != 'RPAREN':
            if self.current_token[0] == 'IDENTIFIER':
                # Look ahead to see if it's a keyword argument (identifier followed by '=')
                if self.peek() and self.peek()[0] == 'ASSIGN':
                    # Keyword argument
                    param_name_token = self.expect('IDENTIFIER')
                    param_name = param_name_token[1]
                    self.expect('ASSIGN')
                    value = self.parse_expression()
                    kwargs[param_name] = value
                else:
                    # Positional argument: parse an expression
                    args.append(self.parse_expression())
            else:
                # Positional argument: parse an expression
                args.append(self.parse_expression())

            # After each argument, expect a comma or closing parenthesis
            if self.current_token and self.current_token[0] == 'COMMA':
                self.expect('COMMA')
            else:
                break

        self.expect('RPAREN')

        return FunctionCall(name=name, args=args, kwargs=kwargs)

    def parse_literal(self):
        """Parse a literal value (number, string, etc.)"""
        if self.current_token:
            if self.current_token[0] == 'STRING':
                str_token = self.expect('STRING')
                # Strip both double and single quotes
                value = str_token[1]
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                return String(value=value)
            elif self.current_token[0] == 'NUMBER':
                num_token = self.expect('NUMBER')
                value = int(num_token[1]) if '.' not in num_token[1] else float(num_token[1])
                return Number(value=value)
            elif self.current_token[0] == 'BOOLEAN':
                bool_token = self.expect('BOOLEAN')
                value = bool_token[1] == 'True'
                return Boolean(value=value)
            elif self.current_token[0] == 'IDENTIFIER' and self.current_token[1] == 'None':
                self.expect('IDENTIFIER')  # 'None'
                return None

        raise SyntaxError(f'Expected literal, got {self.current_token}')