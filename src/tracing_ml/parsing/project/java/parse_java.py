import dataclasses


class Function:
    pass


class Class:
    pass


class Package:
    pass


class Module:
    pass


@dataclasses.dataclass(init=True)
class Project:
    pass


def parse_project(base_dir: str) -> Project:
    """
    As input to the model, the ctx includes
    src
        main
            java
                packageName
                    className
                        fnName
                        fnName
        .
        .
        .
    and additionally, to run each test individually, there must exist an enumeration of all tests.
    :param base_dir:
    :return:
    """
    pass

