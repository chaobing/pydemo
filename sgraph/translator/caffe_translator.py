import json
import logging
import sys
from pathlib import Path
from typing import Any, List, Tuple, Dict

import numpy as np
from google.protobuf import json_format, text_format
from tqdm import tqdm

from sgraph.entity.sgraph import SGraph
from sgraph.entity.snode import *
from sgraph.proto.caffe import caffe_pb2
from sgraph.translator.base_translator import ITranslator
from sgraph.utils import helper
from sgraph.utils.helper import Layout

logger = logging.getLogger(__name__)

class CaffeTranslator(ITranslator):
    """
    Convert a caffe model to graph object
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
        logger.info("Start: caffe_to_graph conversion")
        #print(type(cls))
        model_name, proto_param = cls.__load_raw_model(model_files[0])
        sgraph = cls.__create_sgraph(model_name, proto_param, in_shapes)
        logger.info("End: caffe_to_graph conversion")
        return sgraph

    @classmethod
    def __load_raw_model(cls, model_files: Path) -> (str, dict):
        prototxt: Path = None
        #print(model_files)
        if model_files.suffix == ".prototxt":
            prototxt = model_files
        if prototxt is None:
            logger.error("Not found 'prototxt' file")
            sys.exit(1)
        logger.info("load {0}".format(prototxt))
        proto_net_param = caffe_pb2.NetParameter()
        with open(prototxt, mode="r") as pf:
            text_format.Merge(pf.read(), proto_net_param)

        return prototxt.stem, proto_net_param

    @classmethod
    def __create_sgraph(
        cls, 
        name: str, 
        proto_param,
        in_shapes: Optional[List[List[int]]] = None,
    ) -> SGraph:
        assert name is not None
        assert proto_param is not None
        sgraph = SGraph(name, "caffe")
        logger.info("num. of layers in prototxt: {len(proto_param.layer)}")

        layer_dict: Dict[str, Any] = {}
        for layer in proto_param.layer:
                layer_dict[layer.name] = layer
        for name, layer in tqdm(
            layer_dict.items(),
            total = len(layer_dict),
            desc = "[INFO] parse raw model",
            bar_format="{desc:23}:{percentage:3.0f}%|{bar}{r_bar:50}",
        ):
            op_type = layer.type
            #print(f"source layer info: [type, name] = {op_type}, {name}")
        return sgraph

