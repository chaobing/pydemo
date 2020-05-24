#!/bin/bash
protoc addressbook.proto --python_out=.
dot -Tjpg btree.gv -o btree.jpg
