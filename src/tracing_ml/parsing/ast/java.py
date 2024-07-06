import os.path

from antlr4 import *
from antlr4.tree.Tree import ParseTreeWalker

from tracing_ml.parsing.ast.antlr.java_lexer import JavaLexer
from tracing_ml.parsing.ast.antlr.java_parser import JavaParser


# Custom AST token class
class ASTToken:
    def __init__(self, type, value, line, col):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __str__(self):
        return f'{self.type}_{self.value}_{self.line}_{self.col}'

def load_tokens_dict():
    tokens_dict = {}

    with open(os.path.join(os.path.dirname(__file__), 'antlr', 'JavaParser.tokens'), 'r') as t:
        for token in t.readlines():
            tokens_value = token.rsplit("=", 1)
            token_num = tokens_value[1].strip("\n")
            tokens_dict[int(token_num)] = tokens_value[0]

    return tokens_dict


# Function to parse Java code into AST representation
def load_ast(code):
    input_stream = InputStream(code)
    lexer = JavaLexer(input_stream)
    stream = CommonTokenStream(lexer)  # Custom class to extract tokens from JavaLexer.g4 (optional)
    parser = JavaParser(stream)
    tree = parser.compilationUnit()
    tokens_dict = load_tokens_dict()

    class ASTListener(ParseTreeWalker):
        def __init__(self):
            super().__init__()
            self.ast_tokens = []

        def visitTerminal(self, node):
            self.ast_tokens.append(ASTToken(tokens_dict[node.symbol.type], node.getText(), node.symbol.line, node.symbol.column))

        def visitErrorNode(self, node):
            print(f"Error parsing code: {node.getText()}")

        def enterEveryRule(self, ctx: ParserRuleContext):
            pass

        def exitEveryRule(self, ctx: ParserRuleContext):
            pass

    listener = ASTListener()
    listener.walk(listener, tree)
    return listener.ast_tokens


