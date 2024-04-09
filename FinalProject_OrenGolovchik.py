import re
# Define token types
####################  I didnt change the tokens and left them as they have been before ####################
TOKEN_TYPES = [
    ('IF', r'if'),
    ('THEN', r'then'),
    ('ELSEIF', r'elseif'),
    ('ELSE', r'else'),
    ('END', r'end'),
    ('WHILE', r'while'),
    ('DO', r'do'),
    ('ASSIGN', r'='),
    ('BOOL_OP', r'==|>|<'),
    ('ADD_OP', r'\+|-'),
    ('MUL_OP', r'\*|/'),
    ('NUMBER', r'\d+'),
    ('IDENTIFIER', r'[a-zA-Z][a-zA-Z0-9]*'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('NEWLINE', r'\n'),
    ('SKIP', r'\s+'),
    ('UNKNOWN', r'.')
]
# Compile regular expressions
TOKEN_REGEX = re.compile('|'.join(f'(?P<{token_type}>{pattern})' for token_type, pattern in TOKEN_TYPES))

#lexer function takes the input program as a
# string and returns a list of tokens, where each token is a tuple containing the token type and its value.
def lexer(program):
    tokens = []
    for match in TOKEN_REGEX.finditer(program):
        token_type = match.lastgroup
        value = match.group(token_type)
        if token_type != 'SKIP':
            tokens.append((token_type, value))
    return tokens

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_token = None
        self.token_index = -1
        self.advance()

    def advance(self):
        self.token_index += 1
        if self.token_index < len(self.tokens):
            self.current_token = self.tokens[self.token_index]
        else:
            self.current_token = None

    def parse(self):
        return self.program()

    def program(self):
        statements = []
        while self.current_token:
            statements.append(self.statement())
        return statements

    def statement(self):
        token_type, value = self.current_token
        if token_type == 'IF':
            return self.if_statement()
        elif token_type == 'WHILE':
            return self.while_loop()
        else:
            return self.assignment()

    def if_statement(self):
        self.advance()  # Consume 'if'
        condition = self.boolean_expression()
        self.consume('THEN')
        statement = self.statement()
        self.consume('END')
        return ('if', condition, statement)

    def while_loop(self):
        self.advance()  # Consume 'while'
        condition = self.boolean_expression()
        self.consume('DO')
        statement = self.statement()
        self.consume('END')
        return ('while', condition, statement)

    def assignment(self):
        identifier = self.consume('IDENTIFIER')[1]
        self.consume('ASSIGN')
        expression = self.expression()
        return ('assignment', identifier, expression)

    def boolean_expression(self):
        left = self.expression()
        operator = self.consume('BOOL_OP')[1]
        right = self.expression()
        return (operator, left, right)

    def expression(self):
        term = self.term()
        while self.current_token and self.current_token[0] in ('ADD_OP', 'MUL_OP'):
            operator = self.advance()[1]
            term = (operator, term, self.term())
        return term

    def term(self):
        factor = self.factor()
        while self.current_token and self.current_token[0] in ('MUL_OP',):
            operator = self.advance()[1]
            factor = (operator, factor, self.factor())
        return factor

    def factor(self):
        token_type, value = self.current_token
        if token_type == 'NUMBER':
            self.advance()
            return ('number', value)
        elif token_type == 'IDENTIFIER':
            self.advance()
            return ('identifier', value)
        elif token_type == 'LPAREN':
            self.advance()
            expression = self.expression()
            self.consume('RPAREN')
            return expression

    def consume(self, expected_token_type):
        token_type, value = self.current_token
        if token_type == expected_token_type:
            token = self.current_token
            self.advance()
            return token
        else:
            raise SyntaxError(f'Expected {expected_token_type}, got {token_type}')
class Interpreter:
    def __init__(self, max_result_length=10, max_program_length=100, max_variables=10):
        self.variables = {}
        self.max_result_length = max_result_length
        self.max_program_length = max_program_length
        self.max_variables = max_variables
        self.nested_levels = 0  # Track nested levels

    def evaluate_expression(self, expression):
        try:
            result = eval(expression, {}, self.variables)
            result_str = str(result)
            if len(result_str) > self.max_result_length:
                raise OverflowError("Result exceeds maximum length")
            return result
        except OverflowError:
            print("Overflow error: Result exceeds maximum length")
        except Exception as e:
            print("Error evaluating expression:", e)
        return None

    def execute(self, program):
        if len(program) > self.max_program_length:
            print("Error: Program length exceeds maximum")
            return

        statements = program.split(';')
        for statement in statements:
            if statement.strip():
                self.execute_statement(statement)

    def execute_statement(self, statement):
        tokens = statement.split()

        if tokens[0] == 'if':
            self.nested_levels += 1
            if self.nested_levels <= 3:  # Check if nested levels exceed limit
                expression = ' '.join(tokens[1:tokens.index('then')])
                if self.evaluate_expression(expression):
                    self.execute(' '.join(tokens[tokens.index('then') + 1:tokens.index('end')]))
                else:
                    for i in range(tokens.count('elseif')):
                        start = tokens.index('elseif', i)
                        expression = ' '.join(tokens[start + 1:tokens.index('then', start + 1)])
                        if self.evaluate_expression(expression):
                            self.execute(
                                ' '.join(tokens[tokens.index('then', start + 1) + 1:tokens.index('end', start + 1)]))
                            break
                    else:
                        self.nested_levels -= 1
        elif tokens[0] == 'end':
            self.nested_levels -= 1
        elif tokens[0] == 'while':
            expression = ' '.join(tokens[1:tokens.index('do')])
            self.nested_levels += 1
            while self.evaluate_expression(expression):
                self.execute(' '.join(tokens[tokens.index('do') + 1:tokens.index('end')]))
                self.nested_levels -= 1
                if self.nested_levels <= 3:
                    expression = ' '.join(tokens[1:tokens.index('do')])
                else:
                    break
        else:
            if len(self.variables) >= self.max_variables:
                print("Error: Maximum number of variables exceeded")
                return

            variable = tokens[0]
            if not variable.isidentifier():
                print("Error: Invalid variable name")
                return

            expression = ' '.join(tokens[2:])
            result = self.evaluate_expression(expression)
            if result is not None:
                self.variables[variable] = result

class Variable:
    def __init__(self, name, value=0):
        self.name = name
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

class Statement:
    def __init__(self, interpreter, statement_type, parameters):
        self.interpreter = interpreter
        self.statement_type = statement_type
        self.parameters = parameters

    def execute(self):
        if self.statement_type == "assignment":
            variable_name, expression = self.parameters
            self.interpreter.variables[variable_name].set_value(self.interpreter.evaluate_expression(expression))
        elif self.statement_type == "if_statement":
            condition, then_body, else_body = self.parameters
            if self.interpreter.evaluate_expression(condition):
                self.interpreter.execute(then_body)
            else:
                if else_body:
                    self.interpreter.execute(else_body)
        elif self.statement_type == "while_loop":
            condition, loop_body = self.parameters
            while self.interpreter.evaluate_expression(condition):
                self.interpreter.execute(loop_body)

class Expression:
    def __init__(self, expression):
        self.expression = expression

    def evaluate(self, variables):
        try:
            return eval(self.expression, {}, variables)
        except OverflowError:
            print("Overflow error: Result exceeds maximum length")
        except Exception as e:
            print("Error evaluating expression:", e)
        return None

## Test Cases ##
# Test Case 1: Length exceeds maximum
interpreter = Interpreter(max_result_length=5, max_program_length=50, max_variables=3)
interpreter.execute("x = 1; while x < 10 do if x < 5 then if x < 3 then x = x + 1; end; end; x = x + 1; end;")
print(interpreter.variables)  # Expected: Error: Program length exceeds maximum

# Test Case 2: Maximum number of variables exceeded
interpreter = Interpreter(max_result_length=10, max_program_length=50, max_variables=2)
interpreter.execute("x = 1; y = 2; z = 3;")
print(interpreter.variables)  # Expected: {}

# Test Case 3: Program length exceeds maximum
interpreter = Interpreter(max_result_length=10, max_program_length=20, max_variables=3)
interpreter.execute("x = 1; while x < 10 do x = x + 1; end;")
print(interpreter.variables)  # Expected: {}

# Test Case 4: Result exceeds maximum length
interpreter = Interpreter(max_result_length=5, max_program_length=50, max_variables=3)
interpreter.execute("x = 1000000;")
print(interpreter.variables)  # Expected: {}

## Example Fibonacci ##
# Define variables
limit = 1000
fibonacci_numbers = [0, 1]
prime_fibonacci_numbers = []

# Calculate Fibonacci numbers
next_fibonacci = 0
while next_fibonacci <= limit:
    next_fibonacci = fibonacci_numbers[-1] + fibonacci_numbers[-2]
    if next_fibonacci <= limit:
        fibonacci_numbers.append(next_fibonacci)

# Check if Fibonacci numbers are prime and store prime ones
fib_index = 2
while fib_index < len(fibonacci_numbers):
    num = fibonacci_numbers[fib_index]
    if num > 1:
        is_prime = True
        divisor = 2
        while divisor * divisor <= num:
            if num % divisor == 0:
                is_prime = False
                break
            divisor += 1
        if is_prime:
            prime_fibonacci_numbers.append(num)
    fib_index += 1

## Output prime Fibonacci numbers
print("Prime Fibonacci numbers up to", limit, ":")
prime_index = 0
while prime_index < len(prime_fibonacci_numbers):
    print(prime_fibonacci_numbers[prime_index], end=", ")
    prime_index += 1
print()



## Complex program  ---  NumberProcessor  ##
from abc import ABC, abstractmethod
class Operation(ABC):
    @abstractmethod
    def perform(self, x, y):
        pass


class Addition(Operation):
    def perform(self, x, y):
        return x + y


class Subtraction(Operation):
    def perform(self, x, y):
        return x - y


class Multiplication(Operation):
    def perform(self, x, y):
        return x * y


class Division(Operation):
    def perform(self, x, y):
        if y == 0:
            raise ZeroDivisionError("Error: Division by zero")
        return x / y


class NumberProcessor:
    def __init__(self, numbers):
        self.numbers = numbers

    def process_numbers(self):
        length = len(self.numbers)
        if length == 0:
            raise ValueError("Array is empty")

        if length & 1:  # Check if array length is odd
            middle_index = length // 2
            middle_number = self.numbers[middle_index]
        else:
            middle_index = length // 2 - 1
            middle_number = 0

        X = sum(self.numbers[:middle_index + 1])
        Y = sum(self.numbers[middle_index + 1:])

        X += middle_number

        Z = X * Y
        W = X / Y

        if Z > W:
            return "Z is bigger than W", Z, W
        elif Z < W:
            return "W is bigger than Z", Z, W
        else:
            return "Z and W are equal", Z, W

    def find_max_average_pair(self):
        pairs = []
        index1 = 0
        while index1 < len(self.numbers):
            index2 = index1 + 1
            while index2 < len(self.numbers):
                pairs.append((self.numbers[index1], self.numbers[index2]))
                index2 += 1
            index1 += 1

        max_pair = None
        max_average = float('-inf')
        index = 0
        while index < len(pairs):
            pair = pairs[index]
            average = (pair[0] + pair[1]) / 2
            if average > max_average:
                max_average = average
                max_pair = pair
            index += 1

        index1 = 0
        while index1 < len(pairs):
            index2 = index1 + 1
            while index2 < len(pairs):
                pair1 = pairs[index1]
                pair2 = pairs[index2]
                if (pair1[0] + pair1[1]) / 2 < (pair2[0] + pair2[1]) / 2:
                    temp = pairs[index1]
                    pairs[index1] = pairs[index2]
                    pairs[index2] = temp
                index2 += 1
            index1 += 1

        return pairs


if __name__ == "__main__":
    numbers = [0, 10, 1, 11, 4, 14, 5, 15]

    # Process numbers and compare Z and W
    processor = NumberProcessor(numbers)
    result, Z, W = processor.process_numbers()
    print(result, f"Z = {Z}, W = {W}")

    # Find pairs with the biggest average
    max_average_pairs = processor.find_max_average_pair()
    print("Pairs with the biggest average:")
    max_avg_array = []
    index = 0
    while index < len(max_average_pairs):
        pair = max_average_pairs[index]
        max_avg_array.append((pair, (pair[0] + pair[1]) / 2))
        index += 1
    print(max_avg_array)