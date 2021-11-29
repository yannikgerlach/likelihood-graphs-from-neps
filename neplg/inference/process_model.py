from april.fs import EVENTLOG_DIR
from neplg.inference.coder import EncodingDecodingAttributes

import networkx as nx


class ProcessModel:

    def __init__(self, dataset=None, graph=None):
        self.dataset = dataset
        if graph is None:
            file_name = "graph_" + dataset.dataset_name
            self.graph = self.load_from_gpickle(file_name)
        else:
            self.graph = graph

    @staticmethod
    def load_from_gpickle(file_name):
        file_name += ".gpickle"
        return nx.read_gpickle(EVENTLOG_DIR / file_name)

    def get_coder_attributes(self):
        encoding_mapping = dict()
        for node in self.graph.nodes:
            node_name = self.graph.nodes[node]['value']
            encoding_mapping[node] = node_name
        encoder_decoder_attributes = EncodingDecodingAttributes(decoder=encoding_mapping, coding_per_attribute=False,
                                                                create_encoder=True)
        return encoder_decoder_attributes
