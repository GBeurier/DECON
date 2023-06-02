# a class that schedule job. It manages a cache for data (datacache class) and launch runs (thread for sklearn runs, no thread for gpu runs)
from threading import Thread
import time

from trainingjob import TrainingJob


class Scheduler:
    def __init__(self):
        self.jobs = []
        self.thread_jobs = []
        self.threads = []
        self.is_running = False

    def add_job(self, job: TrainingJob):
        if job.type in ["tensorflow", "pytorch", "keras"]:
            self.jobs.append(job)
        elif job.type == "sklearn":
            self.thread_jobs.append(job)

    def run(self):
        global is_running
        while is_running:
            if not self.thread_jobs:
                time.sleep(0.5)
                continue
            job = self.thread_jobs.pop()
            thread = Thread(target=job.start)  # TODO change this
            thread.start()
            self.threads.append(thread)
            time.sleep(1)

    def start(self):
        self.is_running = True
        self.run()

    def stop(self):
        self.is_running = False
