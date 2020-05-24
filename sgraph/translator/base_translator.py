from pathlib import Path
from typing import List, Optional

from sgraph.entity.sgraph import SGraph
from sgraph.entity.snode import SNode
from sgraph.utils.helper import Layout


class ITranslator(object):
    """
    Translator interface
    """

    @classmethod
    def to_graph(
        cls,
        model_files: List[Path],
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[List[List[int]]] = None,
        *args,
        **kargs
    ) -> SGraph:
        raise NotImplementedError("Translator interface")
