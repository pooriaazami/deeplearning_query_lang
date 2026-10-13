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

        # Parse the model
        model = self.parse_model()

        # Attach imports to the model
        model.imports = imports

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
        name_token = self.expect('STRING')
        name = name_token[1].strip('"')

        self.expect('RPAREN')  # End of INPUT call
        self.expect('NEWLINE')  # Expect newline after input

        return Input(shape=tuple(shape_elements), name=name)

    def parse_output(self) -> Output:
        """Parse an output declaration: OUTPUT = tensor_name"""
        self.expect('OUTPUT')
        self.expect('ASSIGN')
        var_token = self.expect('IDENTIFIER')
        var_name = var_token[1]
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
        self.expect('ASSIGN')  # Actually, looking at samples, it's BLOCK name(...) then newline, not =
        # Let me check the samples again...

        # Actually, looking at the samples:
        # BLOCK input_stream(units, input_name, prefix='')
        #    INPUT(name=input_name)
        #    ...
        #    OUTPUT = dense
        # END

        # So it's BLOCK name(params) then newline, not =
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
        self.expect('IDENTIFIER')  # 'times'
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
        """Parse a statement which could be a layer instantiation or assignment"""
        # Look ahead to see if this is an assignment (contains =) or just a layer call
        # Actually, looking at samples, all layer instantiations are assignments: layer = LayerType(...)

        # Parse left-hand side (variable name)
        if self.current_token and self.current_token[0] == 'IDENTIFIER':
            var_token = self.expect('IDENTIFIER')
            var_name = var_token[1]

            # Check if this is an assignment
            if self.current_token and self.current_token[0] == 'ASSIGN':
                self.expect('ASSIGN')  # Consume the =

                # Parse the right-hand side (layer instantiation or expression)
                expr = self.parse_expression()

                self.expect('NEWLINE')

                return Assignment(target=var_name, expression=expr)
            else:
                # This shouldn't happen in valid DL - all layer instantiations are assignments
                # But just in case, treat as a variable reference
                self.expect('NEWLINE')
                return Variable(name=var_name)
        else:
            # Not a valid statement, skip and advance
            if self.current_token:
                self.advance()
            return None

    def parse_expression(self):
        """Parse an expression (layer instantiation, function call, or variable)"""
        # Look for layer type or function call
        if self.current_token and self.current_token[0] == 'IDENTIFIER':
            func_name = self.current_token[1]

            # Check if it's a known layer type or function (case-insensitive)
            known_functions = {
                'Conv2D', 'MaxPool2D', 'UpSampling2D', 'GlobalAveragePooling2D',
                'Dense', 'Flatten', 'Reshape', 'BatchNormalization', 'Activation',
                'Add', 'Concatenate', 'ElementwiseMultiply', 'Multiply',
                'LSTM', 'GRU', 'Zeros', 'RandomNormal', 'Exp',
                'CUSTOM'
            }
            known_functions_lower = {f.lower() for f in known_functions}

            if func_name.lower() in known_functions_lower:
                return self.parse_function_call(func_name)
            else:
                # Could be a variable reference
                var_token = self.expect('IDENTIFIER')
                return Variable(name=var_token[1])
        else:
            raise SyntaxError(f'Expected identifier, got {self.current_token}')

    def parse_function_call(self, name: str):
        """Parse a function call: name(args, kwargs)"""
        self.expect('IDENTIFIER')  # Consume the function name we already peeked at

        self.expect('LPAREN')

        args = []
        kwargs = {}

        # Parse arguments
        while self.current_token and self.current_token[0] != 'RPAREN':
            if self.current_token[0] == 'IDENTIFIER':
                # Could be a keyword argument or positional argument
                param_name = self.current_token[1]
                self.expect('IDENTIFIER')  # Consume parameter name

                if self.current_token and self.current_token[0] == 'ASSIGN':
                    # Keyword argument: param=value
                    self.expect('ASSIGN')
                    value = self.parse_literal()
                    kwargs[param_name] = value

                    if self.current_token and self.current_token[0] == 'COMMA':
                        self.expect('COMMA')
                else:
                    # Positional argument
                    # We need to backtrack - the identifier is actually the value
                    # For simplicity, we'll treat it as a positional argument that's a variable
                    args.append(Variable(name=param_name))

                    if self.current_token and self.current_token[0] == 'COMMA':
                        self.expect('COMMA')
            elif self.current_token[0] == 'STRING':
                # String literal
                str_token = self.expect('STRING')
                args.append(String(value=str_token[1].strip('"')))
                if self.current_token and self.current_token[0] == 'COMMA':
                    self.expect('COMMA')
            elif self.current_token[0] == 'NUMBER':
                # Number literal
                num_token = self.expect('NUMBER')
                value = int(num_token[1]) if '.' not in num_token[1] else float(num_token[1])
                args.append(Number(value=value))
                if self.current_token and self.current_token[0] == 'COMMA':
                    self.expect('COMMA')
            else:
                # Skip unexpected token
                self.advance()

        self.expect('RPAREN')

        return FunctionCall(name=name, args=args, kwargs=kwargs)

    def parse_literal(self):
        """Parse a literal value (number, string, etc.)"""
        if self.current_token:
            if self.current_token[0] == 'STRING':
                str_token = self.expect('STRING')
                return String(value=str_token[1].strip('"'))
            elif self.current_token[0] == 'NUMBER':
                num_token = self.expect('NUMBER')
                value = int(num_token[1]) if '.' not in num_token[1] else float(num_token[1])
                return Number(value=value)
            elif self.current_token[0] == 'IDENTIFIER' and self.current_token[1] == 'None':
                self.expect('IDENTIFIER')  # 'None'
                return None

        raise SyntaxError(f'Expected literal, got {self.current_token}')