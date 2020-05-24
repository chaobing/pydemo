from collections import namedtuple
from typing import Any, Dict, List, NoReturn, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(lineno)d - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class SNode(object):
    """
    SNode protocol
    """
    def __init__(self, op_name: str, op_type: str):
        assert op_name is not None, "op_name must have a string value"
        self.__op_name = op_name
        self.__op_type = op_type
        self.__bottom = []
        self.__top = []
        self.__in_place = False
        self.__inputs_tensor_shape: List[List[int]] = []
        self.__outputs_tensor_shape: List[List[int]] = []

    @property
    def op_name(self) -> str:
        return self.__op_name
    @op_name.setter
    def op_name(self, name: str) -> NoReturn:
        assert name is not None, "'name' should not be none"
        assert isinstance(name, str), "'name' should be str type"
        self.__op_name = name

    @property
    def op_type(self) -> str:
        return self.__op_type

    @property
    def bottom(self) -> List[str]:
        return self.__bottom
    @bottom.setter
    def bottom(self, bottom: List[str]) -> NoReturn:
        self.__bottom = bottom

    @property
    def top(self) -> List[str]:
        return self.__top
    @top.setter
    def top(self, top: List[str]) -> NoReturn:
        self.__top = top

    @property
    def indegree(self):
        return len(self.__bottom)
    @property
    def outdegree(self):
        return len(self.__top)

    @property
    def is_inplace(self):
        return self.__in_place
    @is_inplace.setter
    def is_inplace(self, flag: bool) -> NoReturn:
        self.__in_place = flag

    @property
    def inputs_tensor_shape(self) -> List[List[int]]:
        return self.__inputs_tensor_shape
    @inputs_tensor_shape.setter
    def inputs_tensor_shape(self, tensors_shape: List[List[int]]) -> NoReturn:
        assert tensors_shape is not None, "'tensors_shape' should not be None"
        self.__inputs_tensor_shape = tensors_shape

    @property
    def outputs_tensor_shape(self) -> List[List[int]]:
        return self.__outputs_tensor_shape
    @outputs_tensor_shape.setter
    def outputs_tensor_shape(self, tensors_shape: List[List[int]]) -> NoReturn:
        assert tensors_shape is not None, "'tensors_shape' should not be None"
        self.__outputs_tensor_shape = tensors_shape

