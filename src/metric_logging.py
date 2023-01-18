import contextlib
import enum
import hashlib
import logging
import multiprocessing
import random
import signal
import time
import typing

import mlflow
import mlflow.entities.metric
import mlflow.exceptions
import numpy as np
import requests.exceptions


class _Message(enum.Enum):
    SYNC = "sync"
    TERMINATE = "terminate"


_MAX_METRICS_PER_BATCH = 1000
_log = logging.getLogger(__name__)


class MetricLogger(contextlib.AbstractContextManager):
    def __init__(
        self,
        experiment_name: typing.Optional[str] = None,
        run_name: typing.Optional[str] = None,
        tags: typing.Optional[typing.Dict[str, typing.Any]] = None,
        description: typing.Optional[str] = None,
        min_sync_metrics: typing.Optional[int] = 1,
        max_sync_metrics: typing.Optional[int] = 2 * _MAX_METRICS_PER_BATCH,
        debug: bool = False,
    ):
        self._log = logging.getLogger(__name__)

        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tags = tags
        self._description = description
        self._metrics_until_sync: typing.Optional[int] = None
        self._debug = debug
        if debug:
            # Immediately sync if debug flag specified
            self._log.debug("Enabling immediate sync")
            min_sync_metrics = max_sync_metrics = 1
        self._min_sync_metrics = min_sync_metrics
        self._max_sync_metrics = max_sync_metrics

        self._active_run: typing.Optional[mlflow.tracking.fluent.ActiveRun] = None
        self._input_queue = multiprocessing.Queue(maxsize=-1)
        self._worker_process: typing.Optional[multiprocessing.Process] = None

        # Avoid global seeding for determining sync frequency
        # Uses same method as Python's random module
        raw_seed = (str(int(time.time() * 1000)) + (run_name or "")).encode()
        self._rng = np.random.default_rng(
            seed=int.from_bytes(raw_seed + hashlib.sha512(raw_seed).digest(), "big")
        )

    def __enter__(self) -> "MetricLogger":
        assert self._active_run is None

        # Make sure experiment exist (synchronous)
        if self._experiment_name is not None:
            _retry(lambda: mlflow.set_experiment(self._experiment_name))

        # Start run
        self._active_run = _retry(
            lambda: mlflow.start_run(
                run_name=self._run_name,
                tags=self._tags,
                description=self._description,
            )
        )

        # Start worker for async metrics logging
        mlflow_client = mlflow.tracking.MlflowClient()
        run_id = self._active_run.info.run_id
        self._worker_process = multiprocessing.Process(
            target=_sync_worker, args=(run_id, mlflow_client, self._input_queue)
        )
        self._worker_process.start()

        self._log.info("Started run with ID %s", run_id)

        return self

    def log_params(self, params: typing.Dict[str, typing.Any]):
        assert self._active_run is not None

        # Synchronous
        _retry(lambda: mlflow.log_params(params))

    def log_tags(self, tags: typing.Dict[str, str]):
        assert self._active_run is not None

        # Synchronous
        _retry(lambda: mlflow.set_tags(tags))

    def log_metric(self, key: str, value: float, step: typing.Optional[int] = None):
        # Asynchronous
        assert self._active_run is not None

        if self._metrics_until_sync is None or self._metrics_until_sync == 0:
            self._metrics_until_sync = self._rng.integers(
                self._min_sync_metrics, self._max_sync_metrics + 1
            )

        # Build and enqueue metric
        timestamp = int(time.time() * 1000)
        metric = mlflow.entities.metric.Metric(key, value, timestamp, step or 0)
        self._input_queue.put(metric, block=False)  # don't block; unlimited size

        # Determine whether to sync stored metrics
        self._metrics_until_sync -= 1
        assert self._metrics_until_sync >= 0
        if self._metrics_until_sync == 0:
            self._input_queue.put(_Message.SYNC, block=False)

    def log_metrics(self, metrics: typing.Dict[str, float], step: typing.Optional[int] = None):
        # Asynchronous
        assert self._active_run is not None

        for key, value in metrics.items():
            self.log_metric(key, value, step)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        assert self._active_run is not None

        # Finish syncing and wait for worker to terminate
        self._input_queue.put(_Message.TERMINATE, block=False)
        self._log.info("Waiting for sync process to finish...")
        self._worker_process.join(timeout=None)  # block until done

        self._log.info("Finishing run logging")
        res = _retry(lambda: self._active_run.__exit__(exc_type, exc_value, traceback))
        self._active_run = None
        self._log.info("Finished run logging")
        return res


def _sync_worker(
    run_id: str, client: mlflow.tracking.MlflowClient, input_queue: multiprocessing.Queue
):
    log = logging.getLogger(__name__)

    # Set up graceful interrupt handling
    class _InterruptHandler(contextlib.AbstractContextManager):
        def __init__(self):
            self._original_handler = None
            self._interrupted = False

        def __enter__(self):
            def _handler(signum, frame):
                self._interrupted = True

            self._original_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _handler)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            signal.signal(signal.SIGINT, self._original_handler)
            return False

        @property
        def interrupted(self) -> bool:
            return self._interrupted

    metrics_backlog = []

    def _sync():
        for idx_offset in range(0, len(metrics_backlog), _MAX_METRICS_PER_BATCH):
            _retry(
                lambda: client.log_batch(
                    run_id=run_id,
                    metrics=metrics_backlog[idx_offset : idx_offset + _MAX_METRICS_PER_BATCH],
                )
            )
        metrics_backlog.clear()

    with _InterruptHandler() as interrupt_handler:
        while True:
            if interrupt_handler.interrupted:
                log.warning("Sync worker received interrupt; clearing queue")
                # Clear and sync queue before terminating after receiving INT
                while input_queue.qsize() > 0:
                    message = input_queue.get(block=False)
                    if isinstance(message, mlflow.entities.metric.Metric):
                        metrics_backlog.append(message)
                input_queue.close()
                _sync()
                break
            else:
                message = input_queue.get(block=True, timeout=None)  # block w/o timeout
                if isinstance(message, mlflow.entities.metric.Metric):
                    metrics_backlog.append(message)
                elif message == _Message.SYNC:
                    _sync()
                elif message == _Message.TERMINATE:
                    input_queue.close()
                    _sync()  # always sync before ending
                    break
                else:
                    assert False


def _retry(
    fn: typing.Callable[[], typing.Any],
    min_backoff: float = 1.0,
    backoff_factor: float = 2.0,
    max_backoff: float = 120.0,
    backoff_jitter: float = 0.2,
) -> typing.Any:
    current_sleep = min_backoff
    while True:
        try:
            return fn()
        except (requests.exceptions.RequestException, mlflow.exceptions.MlflowException) as ex:
            # MLFlow handles retries badly; need this hacky way to check
            if isinstance(ex, mlflow.exceptions.MlflowException):
                # Only exception to retry is if max retries exceeded due to many parallel requests
                if "Max retries exceeded" not in ex.message:
                    raise

            _log.warning("Call failed with exception; sleeping then retrying", exc_info=ex)
            time.sleep(current_sleep)
            current_sleep = min(current_sleep * backoff_factor, max_backoff)
            # Apply some random jitter to avoid all processes retrying at the same time
            current_sleep = random.uniform(-backoff_jitter, backoff_jitter) * current_sleep
            current_sleep = max(min_backoff, current_sleep)
