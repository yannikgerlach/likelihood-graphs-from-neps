import numpy as np
import re


class NextEventPredictor:

    def __init__(self, dataset, model, next_event_threshold, name=None):
        """
            Args:
                next_event_threshold (list of float): List of thresholds, one for each attribute.
        """
        self.dataset = dataset
        self.model = model  # can be a graph or neural network for instance
        self.next_event_threshold = [float(x) for x in re.split(",\s*", next_event_threshold)] if isinstance(next_event_threshold, str) else next_event_threshold
        self.name = self.model.name if name is None else name

    def next_event(self, case, case_length):
        pass

    def get_next_events(self, case, case_length):  # sequence length without padding, not max_len
        return self.next_event(self, case, case_length)

    def event_from_prediction_by_threshold(self, next_event_predictions, attribute):
        """ Given a one-dimensional array of likelihoods (summing up to 1),
            select the next events based on a threshold, i.e. cut off at this threshold.
            The returned events are the indices of the entries satisfying the threshold.
            :return: (events, likelihoods) """
        next_event_attributes = np.where(next_event_predictions >= self.next_event_threshold[attribute])[0]

        if len(next_event_attributes) == 0:
            print("WARNING: No next event for current threshold. Taking the most likely next attribute value.")
            next_event_attributes = [np.argmax(next_event_predictions)]

        next_event_likelihoods = np.take(next_event_predictions, next_event_attributes)

        return next_event_attributes, next_event_likelihoods
