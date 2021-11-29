import numpy as np


class EncodingDecodingAttributes:

    def __init__(self, decoder=None, encoder=None, coding_per_attribute=False,
                 create_decoder=False, create_encoder=False):
        self.coding_per_attribute = coding_per_attribute
        self.encoder = encoder
        self.decoder = decoder
        # create decoder from encoder
        if create_decoder:
            if coding_per_attribute:
                self.decoder = self.inverse_mapping_per_attribute(encoder)
            else:
                self.decoder = self.inverse_mapping(encoder)
        # create encoder from decoder
        if create_encoder:
            if coding_per_attribute:
                self.encoder = self.inverse_mapping_per_attribute(decoder)
            else:
                self.encoder = self.inverse_mapping(decoder)

    @staticmethod
    def inverse_mapping(mapping):
        return {v: k for k, v in mapping.items()}

    def inverse_mapping_per_attribute(self, mapping_per_attribute):
        return [self.inverse_mapping(mapping) for mapping in mapping_per_attribute]

    def encode(self, x, attribute=None):
        """ Encodes a single element. If encoder per attribute, the attribute of x is required (e.g. 0). """
        return self.convert(x=x, attribute=attribute, encode=True)

    def decode(self, x, attribute=None):
        """ Decodes a single element. If decoder per attribute, the attribute of x is required (e.g. 0). """
        return self.convert(x=x, attribute=attribute, encode=False)

    def convert(self, x, attribute=None, encode=True):
        conversion_function = self.encoder if encode else self.decoder
        if self.coding_per_attribute:
            if attribute is None:
                raise Exception('no attribute for encoding')
            return conversion_function[attribute][x]
        return conversion_function[x]

    def encode_sequence(self, sequence, attribute=None):
        return self.convert_sequence(sequence, attribute, True)

    def decode_sequence(self, sequence, attribute=None):
        return self.convert_sequence(sequence, attribute, False)

    def convert_sequence(self, sequence, attribute=None, encode=True):
        """
        :param sequence: The sequence to convert.
        :param attribute: None if no converter per attribute, attribute (e.g. 0) otherwise.
        :param encode: True for encoding the sequence, False for decoding the sequence.
        :return: The converted sequence.
        """
        conversion_function = self.encode if encode else self.decode
        return [conversion_function(x=x, attribute=attribute) for x in sequence]

    def encode_sequence_per_attribute(self, sequence_per_attribute):
        return self.convert_sequence_per_attribute(sequence_per_attribute, encode=True)

    def decode_sequence_per_attribute(self, sequence_per_attribute):
        return self.convert_sequence_per_attribute(sequence_per_attribute, encode=False)

    def convert_sequence_per_attribute(self, sequence_per_attribute, encode=True):
        return [self.convert_sequence(sequence=sequence_per_attribute[attribute], attribute=attribute, encode=encode)
                for (attribute, _) in enumerate(sequence_per_attribute)]

    def encode_sequence_interleaved_attributes(self, sequence_interleaved_attributes, start_attribute=0):
        return self.convert_sequence_interleaved_attributes(sequence_interleaved_attributes, start_attribute, encode=True)

    def decode_sequence_interleaved_attributes(self, sequence_interleaved_attributes, start_attribute=0, with_start_symbol=(False, 1)):
        return self.convert_sequence_interleaved_attributes(sequence_interleaved_attributes, start_attribute, with_start_symbol, encode=False)

    def convert_sequence_interleaved_attributes(self, sequence_interleaved_attributes, start_attribute=0, with_start_symbol=(False, 1), encode=True):
        sequence_length = len(sequence_interleaved_attributes)
        if self.coding_per_attribute:
            number_attributes = self.number_attributes()
            attributes_sequence = np.remainder(np.arange(start_attribute, sequence_length * number_attributes),
                                               number_attributes)
            if with_start_symbol[0]:
                attributes_sequence = np.roll(attributes_sequence, with_start_symbol[1])
                attributes_sequence[0:with_start_symbol[1]] = 0
            return [self.convert(x=value, attribute=attributes_sequence[index], encode=encode)
                    for (index, value) in enumerate(sequence_interleaved_attributes)]
        return [self.convert(x=event, attribute=None, encode=encode) for event in sequence_interleaved_attributes]

    def swap_encoder_decoder(self):
        self.encoder, self.decoder = self.decoder, self.encoder

    def number_attributes(self):
        """ Only produces reasonable output if encoder per attribute. """
        return max(len(self.encoder), len(self.decoder))

    def concatenate_per_attribute(self, coder):
        """ Assumes same amount of attributes and same order of attributes. """
        number_attributes, decoders = self.number_attributes(), []
        for attribute in range(number_attributes):
            decoder = dict()
            for key, value in self.decoder[attribute].items():
                if value in coder.encoder[attribute].keys():
                    decoded_key = coder.encode(value, attribute)
                    decoder[key] = decoded_key
                else:
                    decoder[key] = -1
            decoders.append(decoder)
        return EncodingDecodingAttributes(decoder=decoders, coding_per_attribute=True, create_encoder=True)

    def concatenate_flat(self, coder):
        """ Concatenates two coders that are just a single dictionary, i.e. no dictionary per attribute. """
        decoder_dict = dict()
        # need ground truth integers -> (decode) string -> integer dataset (encode)
        for key, value in self.decoder.items():
            # determine attribute
            attributes = [value in encoder.keys() for encoder in coder.encoder]
            attribute = attributes.index(True)
            decoded_key = coder.encode(value, attribute)
            decoder_dict[key] = decoded_key
        return EncodingDecodingAttributes(decoder=decoder_dict, coding_per_attribute=False, create_encoder=True)

    @staticmethod
    def from_ground_truth_graph(graph, attribute_keys, node_attribute='label'):
        """ Creates an encoder/decoder that maps the nodes to their labels and vice versa. """
        number_attributes = len(attribute_keys)
        graph_walk_decoders = []
        for _ in range(number_attributes):
            graph_walk_decoders.append(dict())
        for node in graph.nodes:
            attribute = attribute_keys.index(graph.nodes[node]['name'])
            graph_walk_decoders[attribute][node] = graph.nodes[node][node_attribute]
        encoder_decoder_attributes = EncodingDecodingAttributes(decoder=graph_walk_decoders, coding_per_attribute=True,
                                                                create_encoder=True)
        return encoder_decoder_attributes

    @staticmethod
    def from_graph(graph, number_attributes, node_attribute='label'):
        """ Creates and encoder/decoder that maps the nodes to their labels and vice versa. """
        graph_walk_decoders = []
        for _ in range(number_attributes):
            graph_walk_decoders.append(dict())
        for node in graph.nodes:
            attribute = graph.nodes[node]['attribute']
            graph_walk_decoders[attribute][node] = graph.nodes[node][node_attribute]
        encoder_decoder_attributes = EncodingDecodingAttributes(decoder=graph_walk_decoders, coding_per_attribute=True,
                                                                create_encoder=True)
        return encoder_decoder_attributes
