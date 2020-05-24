import base64
import json
import logging
import sys
from pathlib import Path, PurePath
from collections import namedtuple
from typing import Any, List, Tuple, Dict, NoReturn, Optional

import numpy as np
from google.protobuf import json_format, text_format
from tqdm import tqdm

from sgraph.entity.sgraph import SGraph
from sgraph.entity.snode import *
from sgraph.proto.tf_pb2 import graph_pb2, tensor_pb2, tensor_shape_pb2
from sgraph.translator.base_translator import ITranslator
from sgraph.utils import helper
from sgraph.utils.helper import Layout

logger = logging.getLogger(__name__)

class TFTranslator(ITranslator):
    """
    Convert a tensorflow model to graph object
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
        logger.info("Start: tensorflow_to_graph conversion")
        print(model_files)
        model_name, raw_nodes = cls.__load_raw_model(model_files)
        sgraph = cls.__create_sgraph(model_name, raw_nodes, in_shapes)
        logger.info("End: tensorflow_to_graph conversion")
        return sgraph

    @classmethod
    def __load_raw_model(cls, model_files: List[Path]) -> (str, dict):
        assert len(model_files) == 1, "model_files should only contain one '.pb' file"
        model_file = model_files[0]
        tfmodel: Path = None
        if model_file.suffix == ".pb":
            tfmodel = model_file
        if tfmodel is None:
            logger.error("Not found '.pb' file")
            sys.exit(1)
        graph_def = graph_pb2.GraphDef()
        with open(tfmodel, "rb") as pf:
            graph_def.ParseFromString(pf.read())
        return tfmodel.stem, graph_def.node

    @classmethod
    def __create_sgraph(
        cls, 
        name: str, 
        layers,
        in_shapes: Optional[List[List[int]]] = None,
    ) -> SGraph:
        assert name is not None
        assert layers is not None
        sgraph = SGraph(name, "tensorflow")
        logger.info("num. of layers in pb: {len(layers)}")
        for layer in layers:
            print(f"{layer.name}, {layer.op}")
        return sgraph
