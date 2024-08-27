import sys, os, re, json, time, datetime, logging, traceback
from typing import Optional
from pathlib import Path

def __func__(offset:int = 0):
    return sys._getframe(1 + offset).f_code.co_name

def __line__(offset:int = 0):
    return sys._getframe(1 + offset).f_lineno

def __info__(msg: Optional[str] = None, offset:int = 0, date:bool = False)->str:
    result = f"[{__func__(1 + offset)}:{__line__(1 + offset)}] {msg}"
    if date:
        result = f"<{datetime.datetime.now()}>" + result
    return result

class UnimplementedError(Exception):
    def __init__(self, msg: Optional[str] = None):
        msg = msg if msg else "Unimplemented method."
        msg = __info__(msg, 1)
        super().__init__(msg)
