import arrow
import flask
from flask import request, jsonify
from functools import partial

from april import Dataset, fs
from april.alignments.binet import BINet
from neplg.api import routines
from neplg.api.jobs import Job, JobStatus
from april.fs import DATE_FORMAT


def get(job_queue):
    """ Creates and returns the flask app. Jobs are put into the given queue. """
    app = flask.Flask(__name__)
    app.config["DEBUG"] = False
    app.config["USE_RELOADER"] = False

    API_NAME = '/api'

    @app.route(f'{API_NAME}/models/train/binet/<int:version>', methods=['POST'])
    def train_model_binet(version):
        """ Creates a job for training a version of the BINet with the given parameters.
            Returns the job id and the location of the future output.
            parameters['event_log']
            parameters['batch_size'] = request.args.get('batch_size', default=500, type=int)
            parameters['validation_split'] = request.args.get('validation_split', default=0.1, type=float)
            parameters['epochs'] = request.args.get('epochs', default=1000, type=int)
            parameters['early_stopping_patience'] = request.args.get('patience', default=5, type=int)
            parameters['early_stopping_delta'] = request.args.get('delta', default=0.01, type=float)"""
        parameters = request.json

        event_log = parameters['event_log']

        # determine output name
        start_time = arrow.now()

        dataset = Dataset(event_log, use_event_attributes=True, use_case_attributes=False)

        (present_activity, present_attribute), combination = routines.get_present_setting(version)
        binet = BINet(dataset, use_event_attributes=True, use_case_attributes=False,
                      use_present_activity=present_activity, use_present_attributes=present_attribute)

        output_name = f'{event_log}_{binet.name}{combination}_{start_time.format(fs.DATE_FORMAT)}'
        output_locations = [output_name]

        # create job
        next_job_id = Job.next_job_id
        job = Job(identifier=next_job_id, output_locations=output_locations,
                  routine=partial(routines.train_binet, output_locations=output_locations, event_log=event_log,
                                  version=version, parameters=parameters))
        job_queue.put_nowait(job)
        Job.next_job_id = next_job_id + 1

        return jsonify({'job_id': next_job_id, 'output_locations': output_locations})

    @app.route(f'{API_NAME}/inference/', methods=['POST'])
    def inference():
        """ Infers a likelihood graph for the given event log using the given next event predictor.
            Returns the job id and the location of the future output.

            Args:
                event_log: The name of the event log.
                model: The name of the model file. Must be either a BINet or a Transformer.
                next_event_threshold: The next event thresholds to be used in the inference.
                    One likelihood threshold for each attribute.
                group_attribute_nodes: Whether to group attribute nodes."""
        parameters = request.json

        event_log, model, next_event_threshold = parameters['event_log'], parameters['model'], parameters['next_event_threshold']
        use_cache, group_attribute_nodes = parameters['cache'], parameters['group_attributes']

        # determine output name
        start_time = arrow.now()
        output_name = f'results_{event_log}-{model}-{start_time.format(DATE_FORMAT)}-{use_cache}-{group_attribute_nodes}-{next_event_threshold}'
        output_locations = [output_name]

        # create job
        next_job_id = Job.next_job_id
        job = Job(identifier=next_job_id, output_locations=output_locations,
                  routine=partial(routines.inference, output_locations=output_locations, event_log=event_log,
                                  model=model, next_event_threshold=next_event_threshold,
                                  group_attribute_nodes=group_attribute_nodes))
        job_queue.put_nowait(job)
        Job.next_job_id = next_job_id + 1

        return jsonify({'job_id': next_job_id, 'output_locations': output_locations})

    @app.route(f'{API_NAME}/evaluate/', methods=['POST'])
    def evaluate():
        parameters = request.json

        event_log, model, next_event_threshold = parameters['event_log'], parameters['model'], parameters['next_event_threshold']  # -1 for automatic
        use_cache, group_attribute_nodes = parameters['cache'], parameters['group_attributes']

        # determine output name
        start_time = arrow.now()
        if next_event_threshold == -1:
            output_name = f'results_{event_log}-{model}-{start_time.format(DATE_FORMAT)}-{use_cache}-{group_attribute_nodes}'
        else:
            output_name = f'results_{event_log}-{model}-{start_time.format(DATE_FORMAT)}-{use_cache}-{group_attribute_nodes}-{next_event_threshold}'
        output_locations = [output_name]

        # create job
        next_job_id = Job.next_job_id
        job = Job(identifier=next_job_id, output_locations=output_locations,
                  routine=partial(routines.evaluate, output_locations=output_locations,
                                  event_log=event_log, model_file=model, next_event_threshold=next_event_threshold,
                                  use_cache=use_cache, group_attribute_nodes=group_attribute_nodes))
        job_queue.put_nowait(job)
        Job.next_job_id = next_job_id + 1

        return jsonify({'job_id': next_job_id, 'output_locations': output_locations})

    @app.route(f'{API_NAME}/jobs/<int:job_id>', methods=['GET'])
    def job_status(job_id):
        """ Returns the current status of the job with the given id.
            Returns unknown if the job with the given id does not exist. """
        # make sure job_id exists
        if job_id < Job.next_job_id:
            return jsonify(Job.status[job_id].name)
        else:
            return jsonify(JobStatus.UNKNOWN.name)

    return app
