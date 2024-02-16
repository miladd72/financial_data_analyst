import random
from time import sleep


def retry(retry_count, logger_print=True, verbose=True, return_last_exception=True):
    def actual_decorator(func):
        def inner(*args, **kwargs):
            current_try = 1
            last_exception = None
            while current_try <= retry_count:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    last_exception = e
                    if verbose:
                        if logger_print:
                            print({"retry_function": {"function": func.__name__, "exception": e.__str__()}})
                        else:
                            print("retry func:{} exception:{} ".format(func.__name__, e.__str__()))
                    current_try += 1
                    retry_delay = random.randint(1, 7)
                    sleep(retry_delay)
                # if current_try == retry_count:
            if logger_print:
                print(({"maximum_retries_reached": {"function": func.__name__,"exception": last_exception.__str__()}}))
            else:
                print("maximum number of retries reached func:{} exception: {}".format(func.__name__, last_exception.__str__()))
            if return_last_exception:
                raise last_exception
            else:
                return None
        return inner
    return actual_decorator
