import sys
from collections import namedtuple
from typing import Dict, List, NoReturn, Optional
import logging
import numpy as np
import graphviz

from sgraph.entity.snode import SNode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(name)s - %(lineno)d - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class SGraph(object):
    """
    SNode protocol
    """
    def __init__(self, name: str, model_type: str):
        if name is None or type(name) is not str or len(name) == 0:
            logger.error("Invalid argument 'name'")
            sys.exit(1)
        if model_type is None or type(model_type) is not str or len(model_type) == 0:
            logger.error("Invalid argument 'model_type'")
            sys.exit(1)
        model_t = model_type.lower()
        if model_t not in ["caffe", "tensorflow", "pytorch"]:
            logger.error("Invalid argument 'model_type', only 'caffe', 'tensorflow', 'pytorch' is surported")
            sys.exit(1)

        self.__name: str = name
        self.__origin: str = model_t
        self.__snodes: List[SNode] = []
        self.__input_snodes: List[SNode] = []
        self.__outputs_snodes: List[SNode] = []
        self.__snode_dict: Dict[str, SNode] = {}

    @property
    def name(self) -> str:
        return self.__name

    @property
    def origin(self) -> str:
        return self.__origin

    @property
    def snodes(self) -> List[SNode]:
        return list(self.__snode_dict.values())
    @snodes.setter
    def snodes(self, snodes: List[SNode]):
        self.__snodes = []
        self.__snode_dict = {}
        for snode in snodes:
            self.update_snode(snode)

    @property
    def input(self) -> List[SNode]:
        return self.__input_snodes

    @property
    def output(self) -> List[SNode]:
        return self.__output_snodes

    def update_snode(self, snode: SNode):
        assert snode is not None, "'snode' should not be None."
        assert isinstance(snode, SNode), "'snode' should be SNode type"
        if self.__snodes is None:
            self.__snodes = []
        self.__snodes.append(snode)

        assert (snode.op_name not in self.__snode_dict), f"insert duplicate node: name: {snode.op_name}, type: {snode.op_type}"
        self.__snode_dict[snode.op_name] = snode

    def toposort(self) -> NoReturn:
        snode_dict: Dict[str, SNode] = {}
        indegree_snode: Dict[str, int] = {}
        zero_nodes = []
        for snode in self.snodes:
            snode_dict[snode.op_name] = snode
            indegree_snode[snode.op_name] = snode.indegree
            if len(snode.bottom) == 0:
                zero_nodes.append(snode)

        sorted_nodes = []
        while len(zero_nodes) > 0:
            snode = zero_nodes.pop(0)
            sorted_nodes.append(snode)
            for cldname in snode.top:
                indegree_snode[cldname] -= 1
                if indegree_snode[cldname] == 0:
                    child = snode_dict.get(cldname)
                    assert(
                        child is not None
                    ), f"Not found snode (name {cldname})."
                    zero_nodes.append(child)
        self.snodes = sorted_nodes

    def get_snode_by_name(self, name: str) -> Optional[SNode]:
        return self.__snode_dict.get(name, None)

    def remove_snode(self, snode: SNode) -> bool:
        if snode.op_name in self.__snode_dict:
            parent_names = snode.bottom
            child_names = snode.top
            if parent_names is not None and len(parent_names) > 0:
                for pname in parent_names:
                    pnode = self.__snode_dict.get(pname)
                    assert pnode is not None, f"Not found snode: (name: {pname})"
                    if len(pnode.top) > 0:
                        idx = pnode.top.index(snode.op_name)
                        if child_names is not None and len(child_names) > 0:
                            if idx == 0:
                                pnode.top = (
                                    [x for x in child_names]
                                    if len(pnode.top) == 1
                                    else [x for x in child_names] + pnode.top[1:]
                                )
                            elif idx == (len(pnode.top) - 1):
                                pnode.top = pnode.top[:idx] + [x for x in child_names]
                            else:
                                pnode.top = (
                                    pnode.top[:idx]
                                    + [x for x in child_names]
                                    + pnode.top[idx + 1:]
                                )
                        else:
                            if idx == 0:
                                pnode.top = [] if len(pnode.top) == 1 else pnode.top[1:]
                            elif idx == len(pnode.top) - 1:
                                pnode.top = pnode.top[:idx]
                            else:
                                pnode.top = pnode.top[:idx] + pnode.top[idx+1:]
            if child_names is not None and len(child_names) > 0:
                for cname in child_names:
                    cnode = self.__snode_dict.get(cname)
                    assert cnode is not None, f"Not found snode: (name: {cname})"
                    if len(cnode.bottom) > 0:
                        idx = cnode.bottom.index(snode.op_name)
                        if parent_names is not None and len(parent_names) > 0:
                            if idx == 0:
                                cnode.bottom = (
                                    [x for x in parent_names]
                                    if len(cnode.bottom) == 1
                                    else [x for x in parent_names] + cnode.bottom[1:]
                                )
                            elif idx == (len(cnode.bottom) - 1):
                                cnode.bottom = cnode.bottom[:idx] + [x for x in parent_names]
                            else:
                                cnode.bottom = (
                                    cnode.bottom[:idx]
                                    + [x for x in parent_names]
                                    + cnode.bottom[idx + 1:]
                                )
                        else:
                            if idx == 0:
                                cnode.bottom = [] if len(cnode.bottom) == 1 else pnode.bottom[1:]
                            elif idx == (len(cnode.bottom) - 1):
                                cnode.bottom = cnode.bottom[:idx]
                            else:
                                cnode.bottom = cnode.bottom[:idx] + pnode.bottom[idx+1:]
            
            del self.__snode_dict[snode.op_name]
            return True
        return False

    def infer_shape(self) -> bool:
        if self.snodes is None or len(self.snodes) == 0:
            return False

    def predecessors(self, snode: SNode) -> List[SNode]:
        pred = []
        if snode.bottom is not None and len(snode.bottom) > 0:
            for pname in snode.bottom:
                pnode = self.get_snode_by_name(pname)
                assert pnode is not None, "Not found predecessor node {pname}"
                pred.append(pnode)
        return pred

    def successors(self, snode: SNode) -> List[SNode]:
        succ = []
        if snode.top is not None and len(snode.top) > 0:
            for cname in snode.top:
                cnode = self.get_snode_by_name(cname)
                assert cnode is not None, "Not found predecessor node {cname}"
                pred.append(cnode)
        return succ

    def num_of_edges(self, snode1: SNode, snode2: SNode) -> int:
        assert snode1 is not None, "'snode1' should not be None."
        assert isinstance(snode1, SNode), "'snode1' should be SNode type."
        assert snode2 is not None, "'snode2' should not be None."
        assert isinstance(snode2, SNode), "'snode2' should be SNode type."
        if snode1 == snode2:
            return 0
        pred1 = self.predecessors(snode1)
        if pred1 and snode2 in pred1:
            return 1
        succ1 = self.successors(snode1)
        if succ1 and snode2 in succ1:
            pred2 = self.predecessors(snode2)
            assert snode1 in pred2
            return 1
        return 0

