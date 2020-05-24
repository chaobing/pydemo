import argparse
import importlib.util
import sys
from pathlib import Path
from typing import List, NoReturn
from sgraph.core import CORE
from sgraph.entity.sgraph import SGraph

__version__ = "0.0.1"

def parse_args():
    desc = """
        sgraph-run --type caffe --layout NHWC --model /path/to/resnet50.proto --out=/path/to/resnet50.info
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--model",
        dest="model_files",
        help="file to model file, (*.pb), (*.prototxt), (get_model.py)",
        action="append",
    )
    parser.add_argument(
        "--type",
        dest="model_type",
        help="type of raw model: 'tensorflow', 'caffe', 'pytorch', default='tensorflow'.",
        default="tensorflow",
    )
    parser.add_argument(
        "--layout",
        dest="layout",
        help="data format used in raw model: 'NHWC' or 'NCHW', default='NHWC'",
        choices=['NHWC', 'NCHW'],
        default="NHWC",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--infer-shape",
        dest="infer_shape",
        help="perform shape inference",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--inputs-shape",
        dest="inputs_shape",
        help="shape of input fmap, 1, 224, 224, 3",
        action="append",
    )
    parser.add_argument(
        "--out",
        dest="out_filename",
        help="file name of output",
        required=True
    )

    args = parser.parse_args()
    return args

def run(args):
    assert args is not None, "'args' should not be None."
    assert args.model_type.lower() in [
        "tensorflow",
        "caffe",
        "pytorch",
    ], "'model_type' should be 'tensorflow' or 'caffe' or 'pytorch'."
    model_t = args.model_type.lower()

    assert args.layout.lower() in [
        "nhwc",
        "nchw",
    ], "'layout' should be 'NHWC' or 'NCHW'."
    layout = args.layout.lower()

    model_files: List[Path] = []
    for file_path in args.model_files:
        model_files.append(Path(file_path))
    assert len(model_files) > 0, "'model_files' should have one or more paths."

    src_dir = model_files[0].parent.absolute()

    if model_t == "pytorch":
        assert(
            args.inputs_shape is not None
        ), "[ERROR] shape info of inputs is need for pytorch, [use --inputs-shape option]."

    in_shapes = []
    if args.inputs_shape:
        assert(
            len(args.inputs_shape) >0
        ), "inputs_shape Error"
        for shape in args.inputs_shape:
            in_shapes.append([int(x.strip()) for x in shape.split(",")])
        print(f"in_shapes: {in_shapes}")
   
    graph: SGraph = CORE.make_graph(
        model_files,
        model_t,
        layout,
        in_shapes=in_shapes if len(in_shapes)>0 else None,
    )

    assert graph is not None, "Graph object is None"
    assert isinstance(graph, SGraph)
    if args.infer_shape:
        graph.infer_shape()

    out_filename = Path(args.out_filename)
#serialize_to_file(graph, out_filename)

def main(args=None):
    args = parse_args()
    print(f"[INFO] {args}")
    run(args)
if __name__ == "__main__":
    sys.exit(main())
