import base64
import importlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, NoReturn

import numpy as np
from google.protobuf import json_format, text_format
from tqdm import tqdm

from sgraph.entity.sgraph import SGraph
from sgraph.entity.snode import SNode
from sgraph.utils import helper
from sgraph.utils.helper import Layout

logger = logging.getLogger(__name__)

class CORE(object):
    @staticmethod
    def make_graph(
        model_files: List[Path],
        model_type: str,
        layout: Layout = Layout.NHWC,
        in_shapes: Optional[List[List[int]]] = None,
    ) -> SGraph:
        """Genearte an Graph instance from model files.
        Parameters
        model_files : List[Path]
        model_type : str
        layout : Layout, optional
        in_shapes: Optional[List[List[int]]]
        Returns
        -------
        Graph
        """
        #print(f"{model_files}, {layout}, {in_shapes}")
        assert model_files is not None, "'model_files' should not be None"
        model_t: str = model_type.lower()
        if model_t not in ['caffe', 'tensorflow', 'pytorch']:
            logger.info(
                "value of 'model_type' should be 'caffe' or 'tensorflow' or 'pytorch'."
            )
            sys.exit(1)
        mode_name: str = model_t + "_translator"
        mod = importlib.import_module("sgraph.translator."+mode_name)
        if mod is None:
            logger.info("{0} translator not found".format(mode_name))
            sys.exit(1)
        
        class_name: str = None
        if model_t == "caffe":
            class_name = "CaffeTranslator"
        elif model_t == "tensorflow":
            class_name = "TFTranslator"
        elif model_t == "pytorch":
            class_name = "PyTorchTranslator"
        else:
            class_name = ""
        logger.debug(f"model_type: {model_t}, translator type: {class_name}")

        #print(class_name)
        translator = None
        if hasattr(mod, class_name):
            translator = getattr(mod, class_name)
        if translator is None:
            logger.info("{0} has no class named {1}".format(mod, class_name))
            sys.exit(1)
        logger.info(f"Start: translate raw model to Graph")
        #print(f"{model_files}, {layout}")
        graph = translator.to_graph(model_files, layout, in_shapes=in_shapes,)
        logger.info(f"End: translate raw model to Graph")
        
        return graph
