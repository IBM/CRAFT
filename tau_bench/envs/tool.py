import abc
from typing import Any, Type, TypeVar

class Tool(abc.ABC):
    @staticmethod
    def invoke(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_info() -> dict[str, Any]:
        raise NotImplementedError

T_Tool = TypeVar("T_Tool", bound=Tool)
ToolsMap = dict[str, T_Tool]