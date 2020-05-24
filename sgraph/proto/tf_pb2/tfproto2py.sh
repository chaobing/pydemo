#!/bin/bash
proto_dir=~/Work/framework/tensorflow/tensorflow/core/framework
proto_list=(
attr_value
function
graph
node_def
op_def
resource_handle
tensor
tensor_shape
types
versions
)

for fname in ${proto_list[@]}
do
  echo "[INFO] coping $fname.proto"
  cp ${proto_dir}/${fname}.proto .
done

sed -i "" "s/tensorflow\/core\/framework\///g" *.proto

protoc *.proto --python_out=.

for fname in ${proto_list[@]}
do
  for imname in ${proto_list[@]}
  do
    echo "[INFO] replace import ${imname} in ${fname}_pb2.py"
    sed -i "" "s/import ${imname}_pb2/import sgraph.proto.tf_pb2.${imname}_pb2/g" ${fname}_pb2.py
  done
done

