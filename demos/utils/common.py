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

import os
import datetime
import sys
import pandas as pd
import logging

logging.basicConfig(format="%(asctime)s %(message)s")
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


def log_info():
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"System: {sys.executable}")

    import jecq

    logger.info(f"Supported instruction sets: {jecq.supported_instruction_sets()}")
    logger.info(f"Jecq version: {jecq.__version__}")
    logger.info(f"Jecq path: {jecq.__path__}")

    build_time = datetime.datetime.fromtimestamp(os.path.getmtime(jecq.__file__))
    logger.info(f"Jecq build time: {build_time}")

    logger.info(f"Jecq module: {jecq.IndexJecq.__module__}")


def df_to_str(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def df_unpack_tuple_column(
    df: pd.DataFrame, tuple_column: str, new_columns: list[str]
) -> pd.DataFrame:
    return df.join(pd.DataFrame(df.pop(tuple_column).tolist(), columns=new_columns))
