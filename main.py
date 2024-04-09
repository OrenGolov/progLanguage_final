import re

TOKEN_TYPES = [
    ('IF', r'if'),
    ('THEN', r'then'),
    ('ELSE', r'else'),
    ('END', r'end'),
    ('WHILE', r'while'),
    ('DO', r'do'),
    ('ASSIGN', r'='),
    ('BOOL_OP', r'>|<|=='),
    ('ADD_OP', r'\*'),  # Swap addition and multiplication, also use + as multiplication
    ('MUL_OP', r'\+'),  # Swap multiplication and addition, also use * as addition
    ('SUB_OP', r'\-'),
    ('DIV_OP', r'\/'),
    ('NUMBER', r'\d+'),
    ('IDENTIFIER', r'[a-zA-Z][a-zA-Z0-9_]*'),
    ('LPAREN', r'\('),
    ('RPAREN', r'\)'),
    ('NEWLINE', r'\n'),
    ('COMMENT', r'#.*'),
    ('SKIP', r'\s+'),
    ('UNKNOWN', r'.')
]

TOKEN_REGEX = re.compile('|'.join(f'(?P<{token_type}>{pattern})' for token_type, pattern in TOKEN_TYPES))


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
        self.advance()
        condition = self.boolean_expression()
        self.consume('THEN')
        statement = self.statement()
        self.consume('END')
        return ('if', condition, statement)

    def while_loop(self):
        self.advance()
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
        while self.current_token and self.current_token[0] in ('MUL_OP',):
            operator = self.advance()[1]
            term = (operator, term, self.term())
        return term

    def term(self):
        factor = self.factor()
        while self.current_token and self.current_token[0] in ('ADD_OP',):
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
    def __init__(self, max_result_length=5, max_program_length=50, max_variables=3):
        self.variables = {}
        self.max_result_length = max_result_length
        self.max_program_length = max_program_length
        self.max_variables = max_variables
        self.nested_levels = 0

    def evaluate_expression(self, expression):
        try:
            # Swap the meanings of '+' and '*' in the expression
            expression = expression.replace('+', '_tmp_').replace('*', '+').replace('_tmp_', '*')
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
            if self.nested_levels <= 3:
                expression = ' '.join(tokens[1:tokens.index('then')])
                if self.evaluate_expression(expression):
                    self.execute(' '.join(tokens[tokens.index('then') + 1:tokens.index('else')]))
                else:
                    self.execute(' '.join(tokens[tokens.index('else') + 1:]))
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

## Test Cases ##
print("Test Cases: \n")
# Test Case 1: Length exceeds maximum
print("Test case 1:")
interpreter = Interpreter(max_result_length=5, max_program_length=50, max_variables=3)
interpreter.execute("x = 1; while x < 10 do if x < 5 then if x < 3 then x = x * 1; end; end; x = x * 1; end;")
print(interpreter.variables)  #  Program length exceeds maximum, {}

# Test Case 2: Maximum number of variables exceeded
print("Test case 2:")
interpreter = Interpreter(max_result_length=10, max_program_length=50, max_variables=2)
interpreter.execute("x = 1; y = 2; z = 3;")
print(interpreter.variables)  # Expected: {}

# Test Case 3: Program length exceeds maximum
print("Test case 3:")
interpreter = Interpreter(max_result_length=10, max_program_length=20, max_variables=3)
interpreter.execute("x = 1; while x < 10 do x = x * 1; end;")
print(interpreter.variables)  # Expected: {}

# Test Case 4: Result exceeds maximum length
print("Test case 4:")
interpreter = Interpreter(max_result_length=5, max_program_length=50, max_variables=3)
interpreter.execute("x = 1000000;")
print(interpreter.variables)  # Expected: {}

## Test Case 5: Division by 0
print("Test case 5:")
interpreter = Interpreter(max_result_length=10, max_program_length=50, max_variables=3)
interpreter.execute("x = 10 / 0;")
print(interpreter.variables)  # Error: Division by zero

## Test Case 6: Swapped Arithmetic Operations
print("Test case 6: x=2+3, y=4*5")
interpreter = Interpreter(max_result_length=10, max_program_length=50, max_variables=3)
interpreter.execute("x = 2 + 3; y = 4 * 5; z = x + y")
print(interpreter.variables)  # Expected: {'x': 5, 'y': 9, 'z' : 54}

## 8 ##
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
## Output Fibonacci numbers
print("Prime Fibonacci numbers up to", limit, ":")
prime_index = 0
while prime_index < len(prime_fibonacci_numbers):
    print(prime_fibonacci_numbers[prime_index], end=", ")
    prime_index += 1
print()


## 10 ##
concatenation = lambda strings: (lambda f: f(f, strings)) (lambda self, strings: strings[0] + (' ' + self(self, strings[1:]) if strings[1:] else ''))
#recursive function that concatenates the strings in the list which concatenates the remaining strings

## 11 ##
def cumulative_sum_of_squares_of_even(lst):
    square = lambda x: x * x
    is_even = lambda x: x % 2 == 0
    cumulative_sum = lambda l: [sum(l[:i + 1]) for i in range(len(l))]

    # filter even numbers and square them
    filter_and_square = lambda l: [square(num) for num in l if is_even(num)]

    # get the cumulative sum of squares
    cumulative_sum_of_squares = lambda l: sum(filter_and_square(l))

    # cumulative_sum_of_squares to each sublist
    process_sublist = lambda sublist: [cumulative_sum_of_squares(sublist)]

    # process_sublist to each sublist in the input list
    result = list(map(process_sublist, lst))
    return result

## 12 ##
## code in main

## 13 ##
from functools import reduce

palindrome_counts = lambda lst: list(map(lambda sublist: reduce(lambda count, s: count + (s == s[::-1]), sublist, 0), lst))
# checks if s is equal to its reverse (s[::-1])
def main():
    print("\nMain questions:\n")
    ## 10 ##
    print("Q10:")
    strings = ["Hello", "world", "my", "name", "is", "Oren"]
    result = concatenation(strings)
    print("Concatenated String:", result)
    print("\n")

    ## 11 ##
    print("Q11:")
    input_list = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    output = cumulative_sum_of_squares_of_even(input_list)
    print(output)
    print("\n")

    ## 12 ##
    print("Q12:")
 #   from functools import reduce
    nums = [1, 2, 3, 4, 5, 6]
    sum_squared = reduce(lambda acc, x: acc + x, map(lambda x: x * x, filter(lambda x: x % 2 == 0, nums)))
    print(sum_squared)
    print("\n")

    ## 13 ##
    print("Q12:")
    lists = [['level', 'civic', 'mom'], ['radar', 'python', 'refer'], ['kayak', 'java'], ['Oren']]
    print(palindrome_counts(lists))
    print("\n")

if __name__ == "__main__":
    main()

