import numpy as np
import itertools

from neplg.inference.next_event_predictors.next_event_prediction import NextEventPredictor


class BINetV1NextEventPredictor(NextEventPredictor):

    def __init__(self, dataset, model, next_event_threshold):
        super().__init__(dataset=dataset, model=model, next_event_threshold=next_event_threshold, name='BINetV1')

    def next_event(self, input_sequence, input_sequence_length):
        """ Given a case, return a set of possible next events. """
        number_attributes = self.dataset.num_attributes
        sequence_length = self.dataset.max_len
        attribute_dimensions = self.dataset.attribute_dims + [1] * number_attributes

        input_case = list(input_sequence[0].astype(int))
        input_case = [x.reshape((1, sequence_length)) for x in input_case]

        prediction = self.model.call(input_case)

        # get prediction for next events
        next_events = []
        for attr in range(number_attributes):
            next_event_predictions = np.array(prediction[attr][:, input_sequence_length-1, :])\
                .reshape((attribute_dimensions[attr]))
            next_events.append(self.event_from_prediction_by_threshold(next_event_predictions, attr))

        next_events_attributes, next_events_likelihoods = zip(*next_events)
        event_combinations = [*itertools.product(*next_events_attributes)]
        likelihood_combinations = [*itertools.product(*next_events_likelihoods)]

        number_next_events = len(event_combinations)
        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            next_event_combinations[0, i, :] = event_combinations[i]
            next_event_combinations[1, i, :] = likelihood_combinations[i]

        return next_event_combinations


class BINetV2NextEventPredictor(NextEventPredictor):

    def __init__(self, dataset, model, next_event_threshold):
        super().__init__(dataset=dataset, model=model, next_event_threshold=next_event_threshold, name='BINetV2')

    # this is for present activity only
    def next_event(self, input_sequence, input_sequence_length):
        number_attributes = self.dataset.num_attributes
        sequence_length = self.dataset.max_len
        attribute_dimensions = self.dataset.attribute_dims + [1] * number_attributes

        sequence = input_sequence

        queue = []

        input_case = [*sequence[0].astype(int)]
        input_case = [x.reshape((1, sequence_length)) for x in input_case]
        prediction = self.model.call(input_case)

        next_event_predictions = np.array(prediction[0][:, input_sequence_length - 1, :]) \
            .reshape((attribute_dimensions[0]))

        next_events = self.event_from_prediction_by_threshold(next_event_predictions, 0)
        next_events = zip(next_events[0], next_events[1])
        # generate new sequence for each possible next event
        for next_event in next_events:
            # skip if prediction for next event is padding
            if next_event[0] == 0:
                continue
            new_sequence = sequence.copy()
            new_sequence[0][0][input_sequence_length] = next_event[0]
            new_sequence[1][0][input_sequence_length - 1] = next_event[1]  # always have one likelihood fewer
            queue.append(new_sequence)

        final_sequences = []

        # need to return a continuation for every attribute
        for sequence in queue:
            input_case = [*sequence[0].astype(int)]
            input_case = [x.reshape((1, sequence_length)) for x in input_case]
            prediction = self.model.call(input_case)

            inner_queue = [sequence]

            # continue with every attribute
            for attribute in range(1, number_attributes):

                next_event_predictions = np.array(prediction[attribute][:, input_sequence_length - 1, :]) \
                    .reshape((attribute_dimensions[attribute]))

                next_events = self.event_from_prediction_by_threshold(next_event_predictions, attribute)
                next_events = zip(next_events[0], next_events[1])
                new_queue = []
                # generate new sequence for each possible next event
                for next_event in next_events:
                    # skip if prediction for next event is padding
                    if next_event[0] == 0:
                        continue
                    for next_sequence in inner_queue:
                        new_sequence = next_sequence.copy()
                        new_sequence[0][attribute][input_sequence_length] = next_event[0]
                        new_sequence[1][attribute][input_sequence_length - 1] = next_event[1]
                        new_queue.append(new_sequence)
                inner_queue = new_queue

            final_sequences.extend(inner_queue)

        number_next_events = len(final_sequences)

        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            sequence = final_sequences[i]
            events = sequence[0][:, input_sequence_length]
            likelihoods = sequence[1][:, input_sequence_length - 1]
            next_event_combinations[0, i, :] = events
            next_event_combinations[1, i, :] = likelihoods

        return next_event_combinations
