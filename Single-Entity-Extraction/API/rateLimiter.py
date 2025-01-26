import multiprocessing
import time
from collections import deque
import datetime
import threading

def log(*args, **kwargs):
    current_process = multiprocessing.current_process().name
    current_thread = threading.current_thread().name
    print(f"{datetime.datetime.now().strftime('[%H:%M:%S]')} [{current_process}-{current_thread}]", *args, **kwargs)

class ThrottleBarrier():
    def __init__(
            self,
            counter: multiprocessing.Value,
            lock: multiprocessing.Lock,
            condition: multiprocessing.Condition,
    ):
        self._counter = counter
        self._lock = lock
        self._condition = condition

    def wait(self):
        with self._condition:
            self._condition.wait()

        with self._lock:
            self._counter.value += 1


class CrossProcessesThrottle():

    def __init__(
            self,
            max_requests: int = 100,
            per_seconds: int = 60,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._max_requests = max_requests
        self._per_seconds = per_seconds
        self._manager = multiprocessing.Manager()
        self._counter = self._manager.Value('b', 0)
        self._lock = self._manager.Lock()
        self._condition = self._manager.Condition()

        self._call_counts = deque(maxlen=per_seconds)
        self._last_counter = 0
        self._last_cycle = 0

    def cycle(self):
        with self._lock:
            counter_value = self._counter.value

        t = time.time()
        t_diff = t - self._last_cycle
        # we use the counter.value to find out how many calls we have made since the last time we checked
        diff = counter_value - self._last_counter
        if diff > 0:
            self._last_counter = counter_value
            for i in range(diff):
                self.call(t)

        current_calls = self.calculate_current_calls(t)

        # remaining calls is the amount of calls we still permit, hence we release that many threads using the condition
        remaining_calls = self._max_requests - current_calls
        with self._condition:
            self._condition.notify(remaining_calls)

        next_slot = self._per_seconds - (t - self._call_counts[0][0]) if len(self._call_counts) else 0

        if t_diff > 5:
            self._last_cycle = t
            log(f"Calls in the last {self._per_seconds} seconds: current={current_calls} :: remaining={remaining_calls} :: total={counter_value} :: next slot in={next_slot:.0f}s")

        time.sleep(0.5)

    def call(self, t):
        if len(self._call_counts) == 0 or t - self._call_counts[-1][0] >= 1:
            # create a new entry if none exists this second
            self._call_counts.append((t, 1))
        else:
            # if there is an entry in the current second, add to the counter of the existing entry
            self._call_counts[-1] = (self._call_counts[-1][0], self._call_counts[-1][1] + 1)

    def calculate_current_calls(self, t):
        # clean up calls that are older than the time limit
        while self._call_counts and t - self._call_counts[0][0] > self._per_seconds:
            self._call_counts.popleft()

        # sum up the calls in the current time window
        current_calls = sum(count for _, count in self._call_counts)
        return current_calls

    def get_barrier(self):
        return ThrottleBarrier(
            counter=self._counter,
            lock=self._lock,
            condition=self._condition,
        )