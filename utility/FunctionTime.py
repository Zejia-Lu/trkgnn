import time
from functools import wraps

# Dictionary to store accumulated time for each function
accumulated_time = {}


def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # Accumulate the elapsed time for the function
        keys = func.__qualname__

        if keys in accumulated_time:
            accumulated_time[keys]['count'] += 1
            accumulated_time[keys]['time'] += elapsed_time
        else:
            accumulated_time[keys] = {
                'count': 1,
                'time': elapsed_time,
            }

        return result

    return wrapper


def print_accumulated_times():
    for function_name, total_time in accumulated_time.items():
        print(f"{function_name} took a total of {total_time['time']:.4f} seconds. [counts: {total_time['count']:4d}]")
