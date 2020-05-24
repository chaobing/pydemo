import sys
from collections import namedtuple
from typing import Any, Dict, List, NoReturn, Optional
import logging
import numpy as np
import graphviz
from graphviz import Digraph, nohtml
import google.protobuf
from tqdm import tqdm
import time
import addressbook_pb2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(levelname)s] - %(name)s - %(lineno)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def test_logger():
    logger.info("start logger")
    logger.fatal(f"hello logger")
    logger.error("end logger")

def test_tqdm():
    logger.info("test_tqdm")
    pbar = tqdm(["a", "b", "c", "d"])
    for char in pbar:
        pbar.set_description("Processing %s" % char)

def test_graph():
    logger.info("test_graph")
    g = Digraph('g', filename='btree.gv', node_attr={'shape': 'record', 'height': '.1'})
    g.node('node0', nohtml('<f0> |<f1> G|<f2>'))
    g.node('node1', nohtml('<f0> |<f1> E|<f2>'))
    g.node('node2', nohtml('<f0> |<f1> B|<f2>'))
    g.node('node3', nohtml('<f0> |<f1> F|<f2>'))
    g.node('node4', nohtml('<f0> |<f1> R|<f2>'))
    g.node('node5', nohtml('<f0> |<f1> H|<f2>'))
    g.node('node6', nohtml('<f0> |<f1> Y|<f2>'))
    g.node('node7', nohtml('<f0> |<f1> A|<f2>'))
    g.node('node8', nohtml('<f0> |<f1> C|<f2>'))
    g.edge('node0:f2', 'node4:f1')
    g.edge('node0:f0', 'node1:f1')
    g.edge('node1:f0', 'node2:f1')
    g.edge('node1:f2', 'node3:f1')
    g.edge('node2:f2', 'node8:f1')
    g.edge('node2:f0', 'node7:f1')
    g.edge('node4:f2', 'node6:f1')
    g.edge('node4:f0', 'node5:f1')
    g

def test_proto():
    logger.info("test_graph")
    address_book = addressbook_pb2.AddressBook()
    person = address_book.people.add()
    person.id = 1
    person.name = "safly"
    person.email = "safly@qq.com"
    person.money = 1000.11
    person.work_status = True
    
    phone_number = person.phones.add()
    phone_number.number = "123456"
    phone_number.type = addressbook_pb2.MOBILE
    maps = person.maps
    maps.mapfield[1] = 1
    maps.mapfield[2] = 2
    #serialize
    serializeToString = address_book.SerializeToString()
    print(serializeToString,type(serializeToString))
    address_book.ParseFromString(serializeToString)
    for person in address_book.people:
      print("p_id{},p_name{},p_email{},p_money{},p_workstatu{}"
            .format(person.id,person.name,person.email,person.money,person.work_status))
      for phone_number in person.phones:
        print(phone_number.number,phone_number.type)
      for key in person.maps.mapfield:
        print(key,person.maps.mapfield[key])

def main():
    test_logger()
    test_tqdm()
    test_graph()
    test_proto()

if __name__ == "__main__":
    sys.exit(main())
