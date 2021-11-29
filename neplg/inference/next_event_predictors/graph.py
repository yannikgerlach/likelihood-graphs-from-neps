import numpy as np

from neplg.inference.next_event_predictors.next_event_prediction import NextEventPredictor


class GraphNextEventPredictor(NextEventPredictor):

    def __init__(self, dataset, model, next_event_threshold):
        super().__init__(dataset=dataset, model=model, next_event_threshold=next_event_threshold, name='Graph')

    def next_event(self, input_case, input_case_length):
        number_attributes = self.dataset.num_attributes
        input_sequence = input_case.copy()
        # determine successors
        queue = [input_sequence]  # might need to be different for start symbol
        # need to return a continuation for every attribute, i.e. we need to
        for i in range(number_attributes):
            new_queue = []
            for sequence in queue:
                sequence_attributes = sequence[0]
                # always get successors of most recent node
                if i == 0:
                    node = sequence_attributes[number_attributes-1][input_case_length - 1]
                else:
                    node = sequence_attributes[i-1][input_case_length]
                # end reached (if not a bug)
                if node == 0:
                    continue
                successors = self.model.successors(node)
                for successor in successors:
                    likelihood = self.model.edges[node, successor]['probability']
                    if likelihood < self.next_event_threshold[i]:
                        continue
                    new_sequence = sequence.copy()
                    new_sequence[0][i][input_case_length] = successor
                    new_sequence[1][i][input_case_length - 1] = likelihood  # always have one likelihood fewer
                    new_queue.append(new_sequence)
            if len(new_queue) == 0:
                new_queue.append(sequence)
            queue = new_queue
        number_next_events = len(queue)
        # extract next attributes and likelihoods from sequences
        next_event_combinations = np.zeros((2, number_next_events, number_attributes), dtype=object)
        for i in range(number_next_events):
            sequence = queue[i]
            events = sequence[0][:, input_case_length]
            likelihoods = sequence[1][:, input_case_length - 1]
            next_event_combinations[0, i, :] = events
            next_event_combinations[1, i, :] = likelihoods
        return next_event_combinations
