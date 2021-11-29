import threading
import logging
import queue

from neplg.api import app

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s', )

# FIFO queue for jobs
job_queue = queue.Queue()


class APIThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(APIThread, self).__init__()
        self.target = target
        self.name = name

    def run(self):
        app.get(job_queue).run()


class WorkerThread(threading.Thread):

    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(WorkerThread, self).__init__()
        self.target = target
        self.name = name

    def run(self):
        while True:
            job = job_queue.get()

            logging.debug(f'executing job with id {job.identifier}, {job_queue.qsize()} remaining jobs in queue')
            job.started()

            job.routine()

            logging.debug(f'finished job with id {job.identifier}')
            job.finished()

            job_queue.task_done()


if __name__ == '__main__':
    p = APIThread(name='api')
    c = WorkerThread(name='worker')

    p.start()
    c.start()
