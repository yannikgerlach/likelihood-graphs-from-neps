import copy
from collections import defaultdict, Counter

import arrow
import networkx as nx
import numpy as np
import pickle

from neplg.inference.coder import EncodingDecodingAttributes
from neplg.inference.nodes import NodeCreator
from april.processmining import EventLog

# special symbols
start_token = "▶"
end_token = "■"
padding_token = "•"


class InferenceResult:

    def __init__(self, dataset_name, graph, inference_time=None,
                 case_generation_time=None, cases_to_graph_conversion_time=None):
        self.dataset_name = dataset_name
        self.graph = graph
        self.next_event_predictor = None
        self.group_attribute_nodes = None
        self.next_event_threshold = None
        self.inference_time = inference_time
        self.case_generation_time = case_generation_time
        self.cases_to_graph_conversion_time = cases_to_graph_conversion_time

    def store(self, file):
        with open(f'{file}.pickle', 'wb') as fh:
            pickle.dump(self.__dict__, fh)

    @staticmethod
    def load(file):
        with open(f'{file}.pickle', 'rb') as fh:
            dict_data = pickle.load(fh)
        result = InferenceResult(dataset_name=None, graph=None)
        result.__dict__.update(dict_data)
        return result


class ExceedSequenceLengthError(Exception):

    def __init__(self):
        super().__init__("The length of a generated case exceeded the maximum sequence length.")


class EmptyLikelihoodGraphError(Exception):

    def __init__(self):
        super().__init__("No cases were generated. The generated likelihood graph is empty.")


def reduce_multiple_edge_values_old(edges, reduce_function):
    """ There can be multiple connections between two nodes with different edge values,
        e.g. edge from 'Develop Method' to 'Experiment' with likelihoods 0.21, 0.34 and 0.26.
        :param edges: Set of entries of the form (source_node, destination_node, edge_value).
        :param reduce_function: A function mapping a list of values to a single value.
        :return: (source_node, destination_node, reduce_function output)."""
    edge_values_per_node_pair = defaultdict(list)
    # gather all likelihoods for a pair of nodes that are connected by edges
    for (from_node, to_node, likelihood) in edges:
        node_likelihoods = edge_values_per_node_pair[(from_node, to_node)]
        node_likelihoods.append(likelihood)
    reduced_edges = set()
    # apply reduce function to each node pair
    for (from_node, to_node), values in edge_values_per_node_pair.items():
        reduced_edges.add((from_node, to_node, reduce_function(values)))
    return reduced_edges


def reduce_multiple_edge_values(edges, reduce_function, graph):
    """ There can be multiple connections between two nodes with different edge values,
        e.g. edge from 'Develop Method' to 'Experiment' with likelihoods 0.21, 0.34 and 0.26.
        :param edges: Set of entries of the form (source_node, destination_node, edge_value).
        :param reduce_function: A function mapping a list of values to a single value.
        :return: (source_node, destination_node, reduce_function output)."""
    edge_values_per_node_pair = defaultdict(list)
    # gather all likelihoods for a pair of nodes that are connected by edges
    for (from_node, to_node, likelihood) in edges:
        paths = generate_paths_until_next_event(graph, to_node)
        transformed_paths = tuple(tuple(path) for path in paths)
        node_likelihoods = edge_values_per_node_pair[(from_node, to_node, transformed_paths)]
        node_likelihoods.append(likelihood)
    reduced_edges = set()
    # apply reduce function to each node pair
    for (from_node, to_node, transformed_paths), values in edge_values_per_node_pair.items():
        reduced_edges.add((from_node, to_node, transformed_paths, reduce_function(values)))
    return reduced_edges


def group_edge_values_per_node(edges, group_by_source_node=True):
    """ :param edges: Set of entries of the form (source_node, destination_node, edge_value).
        :param group_by_source_node: True if group by source_node, otherwise group by destination_node.
        :return: Dictionary where each group by node is associated with a set of edge values. """
    edge_values_per_node = defaultdict(list)
    for (from_node, to_node, _, likelihood) in edges:
        node_likelihoods = edge_values_per_node[from_node] if group_by_source_node else edge_values_per_node[to_node]
        node_likelihoods.append(likelihood)
    return edge_values_per_node


def aggregate_edge_values_per_node(edge_values_per_node, aggregation_function):
    """ :param edge_values_per_node: Dictionary where a node is associated with a set of edge values.
        :param aggregation_function: A function mapping a set of values to a single value.
        :return: A dictionary where each nodes is associated with a single value (output of aggregation function). """
    aggregated_values_per_node = dict()
    for node, values in edge_values_per_node.items():
        aggregated_values_per_node[node] = aggregation_function(values)
    return aggregated_values_per_node


def generate_paths_until_next_event(graph, start_node, original_identifier_map=None):
    """ Generates all succeeding paths of start_node until the next node of attribute 0 in the given graph. """
    completed_walks, queue = [], [[start_node.identifier]]
    while len(queue) > 0:
        walk = queue.pop(0)
        successors = graph.successors(walk[-1])  # most recent node
        for successor in successors:
            new_walk = copy.deepcopy(walk)
            new_walk.append(successor)
            if graph.nodes[successor]['attribute'] == 0:
                completed_walks.append(new_walk)
            else:
                queue.append(new_walk)
    return completed_walks


def normalize_edges(edges, model_graph):
    """ Normalizes the outgoing edges of a (likelihood) graph so that they sum up to 1. """
    # determine outgoing sum for each edge
    edges = reduce_multiple_edge_values(edges, np.average, model_graph)
    edge_values_per_node = group_edge_values_per_node(edges)
    outgoing_sum_per_node = aggregate_edge_values_per_node(edge_values_per_node, sum)
    # divide every edge by the sum
    normalized_edges = set()
    for (from_node, to_node, _, value) in edges:
        node_sum = outgoing_sum_per_node[from_node]
        normalized_value = np.round(value / node_sum, 2)
        normalized_edges.add((from_node, to_node, normalized_value))
    return normalized_edges


def generate_preceding_paths(graph, start_node, original_identifier_map=None):
    """ Generates all preceding paths of start_node in the given graph."""
    completed_walks, queue = [], [[start_node.identifier]]
    while len(queue) > 0:
        walk = queue.pop(0)
        predecessors = graph.predecessors(walk[0])  # most recent node
        for predecessor in predecessors:
            new_walk = copy.deepcopy(walk)
            new_walk.insert(0, predecessor)
            if original_identifier_map is None:
                if graph.nodes[predecessor]['attribute'] == 0 and graph.nodes[predecessor]['value'] in encoded_start_symbols:
                    completed_walks.append(new_walk)
                else:
                    queue.insert(0, new_walk)
            else:
                pred = graph.nodes[predecessor]['value']
                if graph.nodes[predecessor]['attribute'] == 0 and (pred == encoded_start_symbols[0] or \
                        (pred in original_identifier_map.keys() and original_identifier_map[pred] == encoded_start_symbols[0])):
                    completed_walks.append(new_walk)
                else:
                    queue.insert(0, new_walk)
    return completed_walks


def generate_succeeding_paths(graph, start_node, original_identifier_map=None):
    """ Generates all succeeding paths of start_node in the given graph. """
    completed_walks, queue = [], [[start_node.identifier]]
    while len(queue) > 0:
        walk = queue.pop(0)
        successors = graph.successors(walk[-1])  # most recent node
        for successor in successors:
            new_walk = copy.deepcopy(walk)
            new_walk.append(successor)
            if original_identifier_map is None:
                if graph.nodes[successor]['value'] == encoded_end_symbols[0]:
                    completed_walks.append(new_walk)
                else:
                    queue.append(new_walk)
            else:
                succ = graph.nodes[successor]['value']
                if graph.nodes[successor]['attribute'] == 0 and (succ == encoded_end_symbols[0] or \
                        (succ in original_identifier_map.keys() and original_identifier_map[succ] ==
                         encoded_end_symbols[0])):
                    completed_walks.append(new_walk)
                else:
                    queue.insert(0, new_walk)
    return completed_walks


def group_nodes(graph, label_to_nodes, by_predecessors, coder_attributes, number_attributes, identifier_map=None):
    """ Group nodes with the same label based on a node's predecessors or successors in the given graph. """
    grouped_nodes = dict()
    for attribute in label_to_nodes.keys():
        grouped_nodes[attribute] = defaultdict(dict)
        # for every label and the nodes with that label
        for label, nodes in label_to_nodes[attribute].items():
            label_traces = defaultdict(set)
            # get all traces for the nodes with the lab
            for node in nodes:
                # generate predecessors/successors and transform paths
                if by_predecessors:
                    paths = generate_preceding_paths(graph, node, identifier_map)
                    paths = frozenset([tuple(coder_attributes.decode_sequence_interleaved_attributes(s, 0, (True, 1))) for s in paths])
                else:
                    paths = generate_succeeding_paths(graph, node, identifier_map)
                    paths = frozenset([tuple(coder_attributes.decode_sequence_interleaved_attributes(s, attribute, (False, 1))) for s in paths])
                label_traces[paths].add(node)
            # assign numbers to nodes based on their preceding paths
            counter = 0
            for values in label_traces.values():
                for value in values:
                    grouped_nodes[attribute][label][value.identifier] = counter
                counter += 1
    return grouped_nodes


def determine_same_nodes(nodes, attribute=0):
    """ Determines nodes with different identifiers but same labels of the given attribute.
        Note: Should only give a single attribute, labels can be the same for different attributes. """
    same_nodes, label_to_nodes = defaultdict(set), defaultdict(set)
    for node in nodes:
        if node.attributes['attribute'] == attribute:
            label_to_nodes[node.label()].add(node)  # only take 'original' label
    for label, label_nodes in label_to_nodes.items():
        if len(label_nodes) > 1:
            same_nodes[label].update(label_nodes)
    return same_nodes


MINIMUM_NUMBER_CASES = 1000
DISCARDED_PERCENTAGE_THRESHOLD = 0.5


def generate_cases(next_event_predictor, from_model=False, start_symbols=None, end_symbols=None, activity_count_threshold=5):
    """ Generate all possible walks. """
    dataset = next_event_predictor.dataset
    number_attributes, sequence_length = dataset.num_attributes, dataset.max_len

    # create initial sequence with start symbols
    initial_sequence = np.zeros((2, number_attributes, sequence_length), dtype=object)
    initial_sequence[0, :, 0] = start_symbols

    queue, completed_cases, runs = [initial_sequence], [], 0

    number_discarded_cases = 0

    # there are sequences that need to be continued left (create nodes depth-first)
    while len(queue) > 0:
        input_sequence = queue.pop(0)
        # problem: max sequence length already reached, cannot add another event -> discard case
        if max(input_sequence[0][attribute].tolist()[-1] != 0 for attribute in range(number_attributes)):
            number_discarded_cases += 1

            total_number_cases = number_discarded_cases + len(completed_cases)
            percentage_discarded_cases = number_discarded_cases / total_number_cases
            if total_number_cases >= MINIMUM_NUMBER_CASES and percentage_discarded_cases >= DISCARDED_PERCENTAGE_THRESHOLD:
                raise RuntimeError('Too many discarded cases.')

            continue

        # count number of activity occurrences
        input_activities = list(input_sequence[0, 0, :])
        activity_counts = Counter(input_activities)
        del activity_counts[0]  # delete padding counts
        maximum_activity_count = max(activity_counts.values())
        # discard case if an activity is repeated too often
        if maximum_activity_count >= activity_count_threshold:
            number_discarded_cases += 1

            total_number_cases = number_discarded_cases + len(completed_cases)
            percentage_discarded_cases = number_discarded_cases / total_number_cases
            if total_number_cases >= MINIMUM_NUMBER_CASES and percentage_discarded_cases >= DISCARDED_PERCENTAGE_THRESHOLD:
                raise RuntimeError('Too many discarded cases.')

            continue

        # note: this way of determining length might not be needed but is more robust
        input_sequence_length = max(input_sequence[0][attribute].tolist().index(0) for attribute in range(number_attributes))
        # (2, number_next_events, number_attributes)
        next_event_combinations = next_event_predictor.next_event(input_sequence, input_sequence_length)
        number_next_event_combinations = next_event_combinations.shape[1]
        # continue with every possible combination
        for i in range(number_next_event_combinations):
            event_combination = next_event_combinations[0, i, :]
            likelihood_combination = next_event_combinations[1, i, :]
            # if likelihood is 0, discard walks
            if 0 not in event_combination and 0.0 in likelihood_combination:
                continue
            # copy input sequence so that we have a new element that we can extend
            next_input_sequence = input_sequence.copy()
            # add the combination for the next event to the just made copy
            for attribute in range(number_attributes):
                next_input_sequence[0][attribute][input_sequence_length] = event_combination[attribute]
                next_input_sequence[1][attribute][input_sequence_length-1] = likelihood_combination[attribute]
            # check whether end of case is reached
            end_symbols_condition = False
            for attribute in range(number_attributes):  # if any of the attributes reach their end
                if end_symbols[attribute] == event_combination[attribute]:
                    end_symbols_condition = True
                    break
            if from_model:
                # end is reached, add end symbols and append to completed cases
                if end_symbols_condition:
                    # assume that this is the end and set event accordingly, even if only one
                    for attribute in range(number_attributes):
                        next_input_sequence[0][attribute][input_sequence_length] = end_symbols[attribute]
                    completed_cases.append(next_input_sequence)
                elif 0 not in event_combination:
                    queue.insert(0, next_input_sequence)
            else:
                # because we have only one end symbol after the initial walk generation
                if end_symbols_condition:
                    completed_cases.append(next_input_sequence)
                elif 0 not in event_combination:
                    queue.insert(0, next_input_sequence)

    return completed_cases


def convert_cases_to_graph(cases, number_attributes, encoder_decoder_attributes, previous_identifier_map,
                           start_symbols, end_symbols, use_hash_identifier=True, padded_walks=True,
                           predecessor_grouped_nodes=None, successor_grouped_nodes=None):
    """ Converts given cases to a graph, grouping nodes if specified. """

    node_creator = NodeCreator(old_original_identifier_map=previous_identifier_map,
                               encoder_decoder_attributes=encoder_decoder_attributes,
                               predecessor_grouped_nodes=predecessor_grouped_nodes,
                               successor_grouped_nodes=successor_grouped_nodes,
                               use_hash_identifier=use_hash_identifier)

    graph_nodes, graph_edges = set(), set()
    # for every completed walk
    for sequence in cases:
        attribute_sequence, likelihood_sequence = sequence[0], sequence[1]
        # group all values for a sequence entry together
        zipped_attribute_sequence, zipped_likelihood_sequence = [*zip(*attribute_sequence)], [*zip(*likelihood_sequence)]
        # determine length
        first_attribute_sequence = sequence[0][0]
        # if there is some padding
        if first_attribute_sequence[-1] == 0:
            sequence_length = sequence[0][0].tolist().index(0) if padded_walks else len(zipped_attribute_sequence)
        else:
            sequence_length = len(zipped_attribute_sequence)
        # convert single sequence of attributes to set of nodes and edges
        for i in range(sequence_length-1):
            current_event, next_event = zipped_attribute_sequence[i], zipped_attribute_sequence[i+1]
            current_likelihood, next_likelihood = \
                tuple([0] * number_attributes) if i == 0 else zipped_likelihood_sequence[i-1], zipped_likelihood_sequence[i]
            # if start event, add only a single node (instead of one for each attribute) -> do not change start node
            if current_event[0] == start_symbols[0]:
                from_node = node_creator.create_start_node(event=current_event[0], attribute_sequence=attribute_sequence)
                graph_nodes.add(from_node)
            else:
                control_flow_node = node_creator.create_control_flow_node(event=current_event[0], index=i,
                                                                          attribute_sequence=attribute_sequence)
                graph_nodes.add(control_flow_node)

                # connect attribute of control flow node
                from_node = control_flow_node
                for attr in range(number_attributes-1):
                    attribute = attr + 1

                    # use the previous node to condition current node
                    to_node = node_creator.create_attribute_node(event=current_event[attribute], attribute=attribute,
                                                                 index=i, attribute_sequence=attribute_sequence,
                                                                 condition_node=from_node,
                                                                 control_flow_node=control_flow_node)

                    edge_likelihood = current_likelihood[attribute]

                    graph_nodes.add(to_node)
                    graph_edges.add((from_node, to_node, np.round(edge_likelihood, 4)))

                    from_node = to_node

            # add edge of last attr of current event to first of next event
            next_event_identifier = next_event[0]
            # do not change end token
            if next_event_identifier == end_symbols[0]:
                first_node = node_creator.create_end_node(next_event_identifier, attribute_sequence=attribute_sequence)
            else:
                first_node = node_creator.create_control_flow_node(event=next_event_identifier, index=i+1,
                                                                   attribute_sequence=attribute_sequence)

            edge_likelihood = next_likelihood[0]
            graph_nodes.add(first_node)
            graph_edges.add((from_node, first_node, np.round(edge_likelihood, 4)))

    # if there are multiple edges between same nodes (due to likelihood) -> reduce to mean likelihood
    graph_edges = reduce_multiple_edge_values_old(graph_edges, np.mean)

    return graph_nodes, graph_edges, node_creator.new_original_identifier_map


def create_graph(nodes, edges):
    """ Creates a directed graph given nodes and edges. """
    graph = nx.DiGraph()

    for node in nodes:
        graph.add_node(node.identifier, **node.attributes)

    for node_from, node_to, likelihood in edges:
        graph.add_edge(node_from.identifier, node_to.identifier, probability=np.round(likelihood, 4))

    return graph


def get_cases_from_graph(graph, dataset, encoder_decoder_attributes=None):
    """ Generates all possible cases in the given graph.
        :param graph: The graph from which to generate the cases from.
        :param dataset: The dataset which the graph describes.
        :return: A list of cases."""
    number_attributes = dataset.num_attributes

    from neplg.inference.next_event_predictors.graph import GraphNextEventPredictor
    next_event_predictor = GraphNextEventPredictor(dataset=dataset, model=graph, next_event_threshold=[0.0] * number_attributes)

    # use other coder if given one (needed when graph is of different 'structure')
    if encoder_decoder_attributes is None:
        encoder_decoder_attributes = EncodingDecodingAttributes.from_graph(graph, number_attributes, 'label')

    # determine start and end symbols by in and out degree
    start_symbol = [n for n, d in graph.in_degree() if d == 0][0]
    end_symbol = [n for n, d in graph.out_degree() if d == 0][0]

    cases = generate_cases(next_event_predictor=next_event_predictor, from_model=False,
                           start_symbols=[start_symbol]*number_attributes,
                           end_symbols=[end_symbol]*number_attributes)

    return cases, encoder_decoder_attributes, [start_symbol], [end_symbol]


def generate_new_graph_after_iteration(model_graph, dataset, number_attributes, identifier_map):
    print("generating cases from current graph")

    case_generation_graph_start = arrow.now()

    graph_cases, coder_attributes_model, start_symbols, end_symbols = \
        get_cases_from_graph(model_graph, dataset)

    case_generation_from_graph_time = (arrow.now() - case_generation_graph_start).total_seconds()
    print(f'generating cases from current graph took {np.round(case_generation_from_graph_time, 2)} seconds')

    graph_nodes, graph_edges, identifier_map = \
        convert_cases_to_graph(cases=graph_cases, encoder_decoder_attributes=coder_attributes_model,
                               number_attributes=number_attributes, previous_identifier_map=identifier_map,
                               use_hash_identifier=False, padded_walks=True,
                               start_symbols=start_symbols, end_symbols=end_symbols)
    graph = create_graph(graph_nodes, graph_edges)

    # new decoder for converting between labels and graph node identifiers
    new_coder_attributes_model = EncodingDecodingAttributes.from_graph(graph, number_attributes, 'label')

    return graph, graph_nodes, graph_cases, identifier_map, new_coder_attributes_model, start_symbols, end_symbols


def merge_nodes_based_on_paths(model_graph, dataset, base_on_successors, previous_identifier_map, group_attribute_nodes,
                               walks=None):
    number_attributes = dataset.num_attributes

    graph_start = arrow.now()

    if walks is None:
        graph, graph_nodes, graph_cases, identifier_map, coder_attributes_model, start_symbols, end_symbols = \
            generate_new_graph_after_iteration(model_graph=model_graph, dataset=dataset,
                                               number_attributes=number_attributes,
                                               identifier_map=previous_identifier_map)
    else:
        graph, graph_nodes, graph_cases, identifier_map = model_graph, walks[1], walks[0], previous_identifier_map
        coder_attributes_model = EncodingDecodingAttributes.from_graph(graph, number_attributes, 'label')
        start_symbols = [coder_attributes_model.decode([n for n, d in graph.in_degree() if d == 0][0], attribute=0)]
        end_symbols = [coder_attributes_model.decode([n for n, d in graph.out_degree() if d == 0][0], attribute=0)]

    graph_generation_time = (arrow.now() - graph_start).total_seconds()
    print(f"get cases from graph time {np.round(graph_generation_time, 2)} seconds")

    node_start = arrow.now()

    # determine nodes with the same labels, they are candidates for being merged
    same_nodes = dict()
    # depending on setting, group only control flow attribute or all attributes
    if group_attribute_nodes:
        for attribute in range(number_attributes):
            if attribute == 1:
                continue
            same_nodes[attribute] = determine_same_nodes(graph_nodes, attribute=attribute)
    else:
        same_nodes[0] = determine_same_nodes(graph_nodes, attribute=0)

    node_time = (arrow.now() - node_start).total_seconds()
    print(f"determine same nodes times {np.round(node_time, 2)} seconds")

    grouping_start = arrow.now()

    grouped_nodes = group_nodes(graph=graph, label_to_nodes=same_nodes, by_predecessors=not base_on_successors,
                                coder_attributes=coder_attributes_model, identifier_map=previous_identifier_map,
                                number_attributes=number_attributes)

    grouping_time = (arrow.now() - grouping_start).total_seconds()
    print(f"group nodes (based on successors {base_on_successors}) took {np.round(grouping_time, 2)} seconds")

    convert_start = arrow.now()
    if base_on_successors:
        graph_nodes, graph_edges, new_identifier_map = \
            convert_cases_to_graph(cases=graph_cases, number_attributes=number_attributes,
                                   previous_identifier_map=previous_identifier_map,
                                   encoder_decoder_attributes=coder_attributes_model,
                                   successor_grouped_nodes=grouped_nodes,
                                   use_hash_identifier=False, padded_walks=True,
                                   start_symbols=start_symbols, end_symbols=end_symbols)
    else:
        convert_start = arrow.now()

        if walks is None:
            graph_nodes, graph_edges, new_identifier_map = \
                convert_cases_to_graph(cases=graph_cases, number_attributes=number_attributes,
                                       previous_identifier_map=previous_identifier_map,
                                       encoder_decoder_attributes=coder_attributes_model,
                                       predecessor_grouped_nodes=grouped_nodes,
                                       use_hash_identifier=False, padded_walks=True,
                                       start_symbols=start_symbols, end_symbols=end_symbols)
        else:
            graph_nodes, graph_edges, new_identifier_map = \
                convert_cases_to_graph(cases=graph_cases, number_attributes=number_attributes,
                                       previous_identifier_map=None,
                                       encoder_decoder_attributes=coder_attributes_model,
                                       predecessor_grouped_nodes=grouped_nodes,
                                       use_hash_identifier=True, padded_walks=True,
                                       start_symbols=start_symbols, end_symbols=end_symbols)

    convert_time = (arrow.now() - convert_start).total_seconds()
    print(f"convert cases to graph {np.round(convert_time, 2)} seconds")

    return graph_nodes, graph_edges, new_identifier_map, graph_cases


def infer_graph_from_cases(dataset, cases, encoder_decoder_attributes, group_attribute_nodes, padded_walks):
    convert_cases_to_graph_time_start = arrow.now()

    global encoded_start_symbols
    encoded_start_symbols = encoder_decoder_attributes.encode_sequence_interleaved_attributes(
        [EventLog.start_symbol] * dataset.num_attributes, start_attribute=0)

    global encoded_end_symbols
    encoded_end_symbols = encoder_decoder_attributes.encode_sequence_interleaved_attributes(
        [EventLog.end_symbol] * dataset.num_attributes, start_attribute=0)

    # convert cases to graph without grouping nodes
    graph_nodes, graph_edges, original_identifier_map = \
        convert_cases_to_graph(cases=cases, encoder_decoder_attributes=encoder_decoder_attributes,
                               number_attributes=dataset.num_attributes, use_hash_identifier=True,
                               padded_walks=padded_walks, previous_identifier_map=None,
                               start_symbols=encoded_start_symbols, end_symbols=encoded_end_symbols)
    graph = create_graph(graph_nodes, graph_edges)
    number_nodes_edges = (len(graph_nodes), len(graph_edges))

    # group nodes in initial graph based on predecessors
    graph_nodes, graph_edges, original_identifier_map, _ = \
        merge_nodes_based_on_paths(model_graph=graph, dataset=dataset, base_on_successors=False,
                                   previous_identifier_map=original_identifier_map,
                                   walks=(cases, graph_nodes), group_attribute_nodes=group_attribute_nodes)
    graph = create_graph(graph_nodes, graph_edges)

    # alternating group based on preceding and succeeding paths
    initial_iteration = True
    while True:
        iteration_start_time = arrow.now()

        # group based on predecessors (not in the initial iteration however)
        # note: cannot easily put first grouping in loop due to different encoding
        if not initial_iteration:
            graph_nodes, graph_edges, original_identifier_map, _ = \
                merge_nodes_based_on_paths(model_graph=graph, dataset=dataset, base_on_successors=False,
                                           previous_identifier_map=original_identifier_map,
                                           group_attribute_nodes=group_attribute_nodes)
            graph = create_graph(graph_nodes, graph_edges)

        # group based on successors
        graph_nodes, graph_edges, original_identifier_map, graph_walks = \
            merge_nodes_based_on_paths(model_graph=graph, dataset=dataset, base_on_successors=True,
                                       previous_identifier_map=original_identifier_map,
                                       group_attribute_nodes=group_attribute_nodes)
        graph = create_graph(graph_nodes, graph_edges)

        new_number_nodes_edges = (len(graph_nodes), len(graph_edges))

        iteration_time = (arrow.now() - iteration_start_time).total_seconds()
        print(f'iteration took {np.round(iteration_time, 2)} seconds')
        print(f'(nodes, edges): {number_nodes_edges} -> {new_number_nodes_edges}\n')

        # if there is no change to previous iteration, terminate algorithm
        if number_nodes_edges == new_number_nodes_edges:
            break

        number_nodes_edges = new_number_nodes_edges
        initial_iteration = False

    # normalize outgoing edge likelihoods so that they sum up to 1
    graph_edges = normalize_edges(graph_edges, create_graph(graph_nodes, graph_edges))
    graph = create_graph(graph_nodes, graph_edges)

    # time for entire conversion of cases to graph
    cases_to_graph_conversion_time = (arrow.now() - convert_cases_to_graph_time_start).total_seconds()
    print(f"creating a graph from the cases took {np.round(cases_to_graph_conversion_time, 2)} seconds\n")

    return InferenceResult(dataset_name=dataset.dataset_name, graph=graph,
                           cases_to_graph_conversion_time=cases_to_graph_conversion_time)


def infer_graph_using_next_event_predictor(dataset, next_event_predictor, activity_count_threshold,
                                           group_attribute_nodes=False, encoder_decoder_attributes=None,
                                           padded_walks=True):
    graph_generation_start_time = arrow.now()

    # encoding
    if encoder_decoder_attributes is None:
        encoder_decoder_attributes = dataset.get_encoder_decoder_for_attributes()

    global encoded_start_symbols
    encoded_start_symbols = encoder_decoder_attributes.encode_sequence_interleaved_attributes(
        [EventLog.start_symbol] * dataset.num_attributes, start_attribute=0)

    global encoded_end_symbols
    encoded_end_symbols = encoder_decoder_attributes.encode_sequence_interleaved_attributes(
        [EventLog.end_symbol] * dataset.num_attributes, start_attribute=0)

    # generate all possible cases using the given next event predictor
    generate_cases_start = arrow.now()
    print("generating cases using next event predictor")

    model_cases = generate_cases(
        next_event_predictor=next_event_predictor,
        from_model=True,
        start_symbols=encoded_start_symbols,
        end_symbols=encoded_end_symbols,
        activity_count_threshold=activity_count_threshold
    )

    # raise error if no cases generated
    if len(model_cases) == 0:
        raise EmptyLikelihoodGraphError()

    case_generation_time = (arrow.now() - generate_cases_start).total_seconds()
    print(f"generating cases using next event predictor took {np.round(case_generation_time, 2)} seconds\n")

    inference_result = infer_graph_from_cases(
        dataset=dataset,
        cases=model_cases,
        encoder_decoder_attributes=encoder_decoder_attributes,
        padded_walks=padded_walks,
        group_attribute_nodes=group_attribute_nodes,
    )

    # calculate total time for inference time
    inference_time = (arrow.now() - graph_generation_start_time).total_seconds()
    print(f"inferring the graph took {np.round(inference_time, 2)} seconds")

    inference_result.case_generation_time = case_generation_time
    inference_result.inference_time = inference_time

    inference_result.next_event_threshold = next_event_predictor.next_event_threshold
    inference_result.next_event_predictor = next_event_predictor.name
    inference_result.group_attribute_nodes = group_attribute_nodes

    return inference_result
