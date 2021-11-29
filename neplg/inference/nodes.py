from itertools import chain


class NodeCreator:

    def __init__(self, predecessor_grouped_nodes, successor_grouped_nodes, use_hash_identifier,
                 encoder_decoder_attributes, old_original_identifier_map):
        self.predecessor_grouped_nodes = predecessor_grouped_nodes
        self.successor_grouped_nodes = successor_grouped_nodes
        self.use_hash_identifier = use_hash_identifier
        self.encoder_decoder_attributes = encoder_decoder_attributes
        self.old_original_identifier_map = old_original_identifier_map
        self.new_original_identifier_map = dict()

    def create_start_node(self, event, attribute_sequence):
        attribute = 0
        start_node = self.create_control_flow_node(event, 0, attribute_sequence)
        original_label = self.new_original_identifier_map[start_node.identifier]
        start_node_identifier = hash((original_label, attribute))
        self.new_original_identifier_map[start_node_identifier] = self.new_original_identifier_map[start_node.identifier]
        start_node.identifier = start_node_identifier
        start_node.attributes['value'] = event
        start_node.attributes['label'] = original_label
        return start_node

    def create_end_node(self, event, attribute_sequence):
        attribute = 0
        end_node = self.create_control_flow_node(event, 0, attribute_sequence)
        original_label = self.new_original_identifier_map[end_node.identifier]
        end_node_identifier = hash((original_label, attribute))
        self.new_original_identifier_map[end_node_identifier] = self.new_original_identifier_map[end_node.identifier]
        end_node.identifier = end_node_identifier
        end_node.attributes['value'] = event
        end_node.attributes['label'] = original_label
        return end_node

    def create_control_flow_node(self, event, index, attribute_sequence):
        original_identifier = event if self.old_original_identifier_map is None else self.old_original_identifier_map[event]

        node_attributes = dict(value=event, attribute=0, index=index, label=original_identifier, count_uncached=0)
        node = Node(name=event, attributes=node_attributes, attribute_walk=attribute_sequence,
                    use_hash_identifier=self.use_hash_identifier)

        # both cannot happen at the same time
        in_predecessor_grouped_nodes = self.predecessor_grouped_nodes is not None and \
                                       original_identifier in self.predecessor_grouped_nodes[0].keys()

        if in_predecessor_grouped_nodes:
            node.identifier = hash((original_identifier, 0, self.predecessor_grouped_nodes[0][original_identifier][node.identifier]))
            self.new_original_identifier_map[node.identifier] = original_identifier
            return node

        in_successor_grouped_nodes = self.successor_grouped_nodes is not None and \
                                     original_identifier in self.successor_grouped_nodes[0].keys()
        if in_successor_grouped_nodes:
            node.identifier = hash((original_identifier, 0, self.successor_grouped_nodes[0][original_identifier][node.identifier]))
            self.new_original_identifier_map[node.identifier] = original_identifier
            return node

        self.new_original_identifier_map[node.identifier] = original_identifier
        return node

    def create_attribute_node(self, event, attribute, index, attribute_sequence, condition_node, control_flow_node):

        original_identifier = event if self.old_original_identifier_map is None else self.old_original_identifier_map[event]

        node_attributes = dict(value=original_identifier, attribute=attribute, index=index, label=original_identifier, count_uncached=0)
        node = Node(name=original_identifier, attributes=node_attributes, attribute_walk=attribute_sequence,
                    use_hash_identifier=self.use_hash_identifier)

        # if not grouped, handle as normal -> condition on node
        new_identifier = hash((original_identifier, attribute, condition_node.identifier))
        grouped_nodes_key = hash((original_identifier, attribute, condition_node.ungrouped_identifier))
        node.condition_node = condition_node
        node.ungrouped_identifier = grouped_nodes_key
        node.identifier = new_identifier

        # both cannot happen at the same time
        in_predecessor_grouped_nodes = self.predecessor_grouped_nodes is not None and \
                                       self.predecessor_grouped_nodes.get(attribute) is not None and \
                                       original_identifier in self.predecessor_grouped_nodes[attribute].keys()

        if in_predecessor_grouped_nodes:
            node.identifier = hash((original_identifier, attribute,
                                    self.predecessor_grouped_nodes[attribute][original_identifier][grouped_nodes_key]))
            self.new_original_identifier_map[node.identifier] = original_identifier
            return node

        in_successor_grouped_nodes = self.successor_grouped_nodes is not None and \
                                     self.successor_grouped_nodes.get(attribute) is not None and \
                                     original_identifier in self.successor_grouped_nodes[attribute].keys()
        
        if in_successor_grouped_nodes:
            node.identifier = hash((original_identifier, attribute,
                                    self.successor_grouped_nodes[attribute][original_identifier][grouped_nodes_key]))
            self.new_original_identifier_map[node.identifier] = original_identifier
            return node

        self.new_original_identifier_map[node.identifier] = original_identifier
        return node


class Node:

    def __init__(self, name, attributes, attribute_walk, use_hash_identifier=True):
        self.attributes = attributes
        sequence = tuple(chain.from_iterable(zip(*attribute_walk)))  # do not need later
        if 0 in sequence:
            sequence = sequence[:sequence.index(0)]  # remove padding
        self.ungrouped_identifier = None
        self.sequence = sequence
        self.condition_node = None
        if use_hash_identifier and attributes['attribute'] == 0:
            self.identifier = hash((name, sequence, self.attributes['index'], self.attributes['attribute']))
            self.ungrouped_identifier = self.identifier
        else:
            self.identifier = hash((name, self.attributes['attribute']))
            self.ungrouped_identifier = self.identifier

    def label(self):
        if 'label' in self.attributes.keys():
            return self.attributes['label']
        return self.identifier

    def __hash__(self):
        return self.identifier

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.identifier == other.identifier
        if isinstance(other, int):
            return self.identifier == other
        return NotImplemented


class NodeHash:

    def __init__(self, identifier):
        self.identifier = identifier

    def __hash__(self):
        return self.identifier


class Edge:

    def __init__(self, source, destination, value):
        self.source = source
        self.destination = destination
        self.value = value

    def __hash__(self):
        return hash((self.source, self.destination, self.value))

    def __eq__(self, other):
        eq_source = self.source == other.source
        if not eq_source:
            return False
        eq_destination = self.destination == other.destination
        if not eq_destination:
            return False
        eq_value = self.value == other.value
        if not eq_value:
            return False
        return True

