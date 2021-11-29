from april.processmining import EventLog

from itertools import chain


def flatten_walk(walk):
    """ Interleaves a walk, i.e. arranges arrays for each attribute so that positions are grouped.
        Example: [[Identify Problem, Related Work], [Supervisor, Student]] -> [Identify Problem, Supervisor, Related..]
        :param walk: An array of arrays for each attribute, i.e. [[] * number_attributes] """
    return [*chain.from_iterable(zip(*walk))]


def transform_event_walks_two_coders(walks, encoder_decoder_attributes1, encoder_decoder_attributes2,
                                     padded_walks=True, with_start_symbol=(False, 1), remove_events=(0, 0)):
    transformed_walks = []
    for walk in walks:
        walk_per_event = [*chain.from_iterable(zip(*walk[0]))]
        # cut off padding
        if padded_walks:
            walk_per_event_length = walk_per_event.index(0)
            walk_per_event = walk_per_event[0:walk_per_event_length]
        walk_per_event = encoder_decoder_attributes1.decode_sequence_interleaved_attributes(walk_per_event, with_start_symbol=with_start_symbol)
        walk_per_event = encoder_decoder_attributes2.decode_sequence_interleaved_attributes(walk_per_event, with_start_symbol=with_start_symbol)
        # cut off at end symbol if present in walk
        if EventLog.end_symbol in walk_per_event:
            end_symbol_index = walk_per_event.index(EventLog.end_symbol)
            walk_per_event = walk_per_event[:end_symbol_index+1]
        walk_per_event = walk_per_event[remove_events[0]:-remove_events[1]]
        walk_per_event_string = tuple(walk_per_event)
        transformed_walks.append(walk_per_event_string)
    return transformed_walks


def transform_event_walks(walks, encoder_decoder_attributes, padded_walks=True, with_start_symbol=(False, 1),
                          remove_events=(0, 0), output_format=tuple):
    transformed_walks = []
    for walk in walks:
        walk_per_event = [*chain.from_iterable(zip(*walk[0]))]
        # cut off padding
        if padded_walks:
            walk_per_event_length = walk_per_event.index(0) if walk_per_event[-1] == 0 else len(walk_per_event)
            walk_per_event = walk_per_event[0:walk_per_event_length]
        if encoder_decoder_attributes:
            walk_per_event = encoder_decoder_attributes.decode_sequence_interleaved_attributes(walk_per_event, with_start_symbol=with_start_symbol)
        # cut off at end symbol if present in walk
        if EventLog.end_symbol in walk_per_event:
            end_symbol_index = walk_per_event.index(EventLog.end_symbol)
            walk_per_event = walk_per_event[:end_symbol_index+1]
        if remove_events[1] == 0:
            walk_per_event = walk_per_event[remove_events[0]:]
        else:
            walk_per_event = walk_per_event[remove_events[0]:-remove_events[1]]
        walk_per_event_string = output_format(walk_per_event)
        transformed_walks.append(walk_per_event_string)
    return transformed_walks


def transform_event_walk(walk_attributes, encoder_decoder_attributes, padded_walks=True, with_start_symbol=(False, 1),
                         remove_events=(0, 0)):
    walk_per_event = [*chain.from_iterable(zip(*walk_attributes))]
    # cut off padding
    if padded_walks:
        walk_per_event_length = walk_per_event.index(0)
        walk_per_event = walk_per_event[0:walk_per_event_length]
    walk_per_event = encoder_decoder_attributes.decode_sequence_interleaved_attributes(walk_per_event, with_start_symbol=with_start_symbol)
    # cut off at end symbol if present in walk
    if EventLog.end_symbol in walk_per_event:
        end_symbol_index = walk_per_event.index(EventLog.end_symbol)
        walk_per_event = walk_per_event[:end_symbol_index+1]
    walk_per_event = walk_per_event[remove_events[0]:-remove_events[1]]
    walk_per_event_string = tuple(walk_per_event)
    return walk_per_event_string


def transform_event_walk_two_coder(walk_attributes, encoder_decoder_attributes1, encoder_decoder_attributes2,
                                   padded_walks=True, with_start_symbol=(False, 1), remove_events=(0, 0)):
    walk_per_event = [*chain.from_iterable(zip(*walk_attributes))]
    # cut off padding
    if padded_walks:
        walk_per_event_length = walk_per_event.index(0)
        walk_per_event = walk_per_event[0:walk_per_event_length]
    walk_per_event = encoder_decoder_attributes1.decode_sequence_interleaved_attributes(walk_per_event, with_start_symbol=with_start_symbol)
    walk_per_event = encoder_decoder_attributes2.decode_sequence_interleaved_attributes(walk_per_event, with_start_symbol=with_start_symbol)
    # cut off at end symbol if present in walk
    if EventLog.end_symbol in walk_per_event:
        end_symbol_index = walk_per_event.index(EventLog.end_symbol)
        walk_per_event = walk_per_event[:end_symbol_index+1]
    walk_per_event = walk_per_event[remove_events[0]:-remove_events[1]]
    walk_per_event_string = tuple(walk_per_event)
    return walk_per_event_string


def transform_event_walk_without_decoding(walks, padded_walks=True, remove_events=(0, 0)):
    transformed_walks = []
    for walk_attributes in walks:
        walk_per_event = [*chain.from_iterable(zip(*walk_attributes))]
        # cut off padding
        if padded_walks and 0 in walk_per_event:
            walk_per_event_length = walk_per_event.index(0)
            walk_per_event = walk_per_event[0:walk_per_event_length]
        if remove_events[1] != 0:
            walk_per_event = walk_per_event[remove_events[0]:-remove_events[1]]
        walk_per_event_string = tuple(walk_per_event)
        transformed_walks.append(walk_per_event_string)
    return transformed_walks


def transform_walk_without_decoding(walk_attributes, padded_walks=True, remove_events=(0, 0)):
    walk_per_event = [*chain.from_iterable(zip(*walk_attributes))]
    # cut off padding
    if padded_walks and 0 in walk_per_event:
        walk_per_event_length = walk_per_event.index(0)
        walk_per_event = walk_per_event[0:walk_per_event_length]
    if remove_events[1] != 0:
        walk_per_event = walk_per_event[remove_events[0]:-remove_events[1]]
    walk_per_event_string = tuple(walk_per_event)
    return walk_per_event_string


def transform_event_attribute_walks(walks, encoder_decoder_attributes):
    transformed_walks = []
    for walk in walks:
        walk_list = [[x for x in w[0] if x != 0] for w in walk]
        walk_per_event_string = transform_event_walk(walk_list, encoder_decoder_attributes)
        transformed_walks.append(walk_per_event_string)
    return transformed_walks


def transform_walk_to_dict_key(walk, padded_walks):
    """ Creates an interleaved walk as a string. Can be used as dictionary key for instance.
        Note: The individual walk attributes must all be of same length.
        :param walk: list of walk attributes
        :param padded_walks: Whether the walks are padded.
        :return: """
    walk_per_event = [*chain.from_iterable(zip(*walk))]
    # remove padding (only if there is padding)
    if padded_walks and walk_per_event[-1] == 0:
        walk_per_event_length = walk_per_event.index(0)
        walk_per_event = walk_per_event[0:walk_per_event_length]
    transformed_walk = tuple(walk_per_event)
    return transformed_walk
