import networkx as nx
import tensorflow as tf

import numpy as np
import neplg
from april.anomalydetection import Binarizer
from april.anomalydetection.binet.core import binet_predictions_scores
from april.enums import Heuristic, Strategy
from neplg.inference import inference, drawing
from neplg.inference.inference import InferenceResult
from april.fs import MODEL_DIR,  EVALUATION_DIR, EVENTLOG_DIR
from april import Dataset, fs
from april.alignments.binet import BINet
from neplg.inference.next_event_predictors.binet import BINetV1NextEventPredictor, BINetV2NextEventPredictor


def get_present_setting(version):
    present_settings = {1: (False, False), 2: (True, False), 3: (True, True)}
    combination = {1: '00', 2: '10', 3: '11'}
    return present_settings[version], combination[version]


def train_binet(output_locations, event_log, version, parameters):
    """ Train a BINet with the given parameters. """
    dataset = Dataset(event_log, use_event_attributes=True, use_case_attributes=False)

    (present_activity, present_attribute), _ = get_present_setting(version)
    binet = BINet(
        dataset,
        use_event_attributes=True,
        use_case_attributes=False,
        use_present_activity=present_activity,
        use_present_attributes=present_attribute
    )

    x, y = dataset.features, dataset.targets
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    binet.fit(
        x=x,
        y=y,
        batch_size=parameters['batch_size'],
        epochs=parameters['epochs'],
        validation_split=parameters['validation_split'],
        callbacks=[callback]
    )

    output_name = output_locations[0]
    binet.save_weights(str(fs.MODEL_DIR / f'{output_name}.h5'))

    tf.keras.backend.clear_session()


def get_binet_model(dataset, model, use_present_activity, use_present_attributes):
    """ Loads the specified BINet model. """
    binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                  use_present_activity=use_present_activity, use_present_attributes=use_present_attributes)
    binet([f[:1] for f in dataset.features])
    binet.load_weights(str(MODEL_DIR / model) + ".h5")
    return binet


def graph_add_display_names(graph, coder_attributes):
    """ Add display names to a graph.
        :param graph: The graph for which the display names are to be added.
        :param coder_attributes: Decoding labels to display name. """
    # add identifier attribute
    for node in graph.nodes:
        node_attributes = graph.nodes[node]
        identifier = coder_attributes.decode(node_attributes['label'], node_attributes['attribute'])
        node_attributes['display_name'] = identifier


def inference(output_locations, event_log, model, threshold_heuristic, next_event_threshold, group_attribute_nodes, activity_count_threshold):
    dataset = Dataset(event_log, use_event_attributes=True)
    file_name = output_locations[0]

    # create next event predictor from given model
    use_present_activity, use_present_attributes = [bool(int(o)) for o in model.split('_')[1][-2:]]
    binet = get_binet_model(dataset, model, use_present_activity, use_present_attributes)

    if threshold_heuristic != 'manual':
        heuristic = {'mean': Heuristic.LP_MEAN, 'left': Heuristic.LP_LEFT, 'right': Heuristic.LP_RIGHT}[threshold_heuristic]

        predictions, scores = binet_predictions_scores(
            model=binet,
            dataset=dataset,
            features=dataset.features
        )

        binarizer = Binarizer(
            predictions=predictions,
            scores=scores,
            mask=~dataset.mask,
            features=dataset.flat_features
        )

        likelihood_thresholds = binarizer.get_tau(
            scores=scores,
            heuristic=heuristic,
            strategy=Strategy.ATTRIBUTE
        )
        next_event_threshold = [np.round(x, 4) for x in abs(likelihood_thresholds - 1)]
        print('using thresholds', next_event_threshold)
        output_locations[0] = output_locations[0] + '-'.join([str(x) for x in next_event_threshold])
        file_name = output_locations[0]

    # either BINetV1 or BINetV2
    model_class = BINetV2NextEventPredictor if use_present_activity else BINetV1NextEventPredictor
    next_event_predictor = model_class(
        dataset=dataset,
        model=binet,
        next_event_threshold=next_event_threshold
    )

    # infer graph and store results
    inference_results = neplg.inference.inference.infer_graph_using_next_event_predictor(
        dataset=dataset,
        next_event_predictor=next_event_predictor,
        group_attribute_nodes=group_attribute_nodes,
        activity_count_threshold=activity_count_threshold
    )

    inference_results.store(EVALUATION_DIR / file_name)
    drawing.draw_and_store_likelihood_graph_with_colors(
        graph=inference_results.graph,
        dataset=dataset,
        file_name=file_name,
        coder_attributes=dataset.get_encoder_decoder_for_attributes()
    )


def evaluate(output_locations, event_log, file_name):
    dataset = Dataset(event_log, use_event_attributes=True)

    inference_result = InferenceResult.load(EVALUATION_DIR / file_name)

    from neplg.inference.evaluation import evaluate as evaluate_result
    ground_truth_process_model = nx.read_gpickle(EVENTLOG_DIR / f'graph_{dataset.dataset_name}.gpickle')
    evaluate_result(name=output_locations[0], dataset=dataset,
                    ground_truth_process_model=ground_truth_process_model,
                    inference_result=inference_result)
