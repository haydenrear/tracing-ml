import dataclasses
import typing

from tracing_ml.parsing.ast.java import ASTToken

AstNodeT = typing.ForwardRef("AstNode")
ExecAstNodeT = typing.ForwardRef("ExecAstNode")


@dataclasses.dataclass(init=True)
class AstNode:
    next: list[AstNodeT]


@dataclasses.dataclass(init=True)
class Ast:
    root: AstNode


@dataclasses.dataclass(init=True)
class ExecPassDescr:
    num_passes: int


@dataclasses.dataclass(init=True)
class ExecAstNode:
    ast_node: AstNode
    exec_pass: ExecPassDescr
    next: list[ExecAstNodeT]


@dataclasses.dataclass(init=True)
class ExecAst:
    root: ExecAstNode


def from_antlr(tokens: list[ASTToken]) -> Ast:
    pass
