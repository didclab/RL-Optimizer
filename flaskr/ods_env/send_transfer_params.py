import os

import pika


class TransferParams(object):
    def __init__(self, cc, p, pp, chunk_size, node_name):
        self.cc = cc
        self.pp = pp
        self.p = p
        self.chunk_size = chunk_size
        self.transfer_node_name = node_name

class SendTransferParams:
    def __init__(self, user, pwd):
        pass

    def send_params(self, queue_name, transfer_params=TransferParams):
        pass
