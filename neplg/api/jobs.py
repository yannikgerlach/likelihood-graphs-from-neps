from enum import Enum


class JobStatus(Enum):
    WAITING = 'waiting'
    STARTED = 'started'
    FINISHED = 'finished'
    UNKNOWN = 'unknown'


class Job:

    next_job_id = 0
    status = {}

    def __init__(self, identifier, output_locations, routine):
        self.identifier = identifier
        self.output_locations = output_locations
        self.routine = routine
        self.waiting()

    def waiting(self):
        Job.status[self.identifier] = JobStatus.WAITING

    def started(self):
        Job.status[self.identifier] = JobStatus.STARTED

    def finished(self):
        Job.status[self.identifier] = JobStatus.FINISHED
