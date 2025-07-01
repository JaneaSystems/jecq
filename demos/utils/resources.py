# Copyright (c) 2025 Janea Systems
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import time
import gc
import os
import psutil

import logging

logger = logging.getLogger(__name__)


def get_memory_usage() -> float:
    if gc.isenabled():
        gc.collect()

    pid = os.getpid()
    process = psutil.Process(pid)
    is_posix = os.name == "posix"
    mem = process.memory_full_info() if is_posix else process.memory_info()

    usage = mem.uss if is_posix else mem.rss

    return round(usage / (1024 * 1024), 2)


def log_memory_usage(label: str = ""):
    usage = get_memory_usage()
    usage_str = f"Total memory usage : {usage} MB"

    if label:
        logger.info(f"[{label}] {usage_str}")
    else:
        logger.info(usage_str)


def time_it(func) -> float:
    start_time = time.perf_counter()
    func()
    end_time = time.perf_counter()
    return round(end_time - start_time, 2)
