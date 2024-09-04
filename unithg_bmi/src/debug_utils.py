import sys, os, re, json, time, datetime, logging, traceback
from typing import Optional
from pathlib import Path

# debug_utils.py
# This file provides utilities for debugging and tracing code execution.

def __func__(offset:int = 0):
    """
    Return the name of the function offset frames up the call stack. (offset: 0 = current function, 1 = caller function)
    """
    return sys._getframe(1 + offset).f_code.co_name

def __line__(offset:int = 0):
    """
    Return the line number of the function call offset frames up the call stack. (offset: 0 = current function, 1 = caller function)
    """
    return sys._getframe(1 + offset).f_lineno

def __info__(msg: Optional[str] = None, offset:int = 0, date:bool = False)->str:
    """
    Return a string with the function name, line number, and message. (offset: 0 = current function, 1 = caller function)
    """
    result = f"[{__func__(1 + offset)}:{__line__(1 + offset)}] {msg}"
    if date:
        result = f"<{datetime.datetime.now()}>" + result
    return result

class UnimplementedError(Exception):
    """
    Custom exception for unimplemented methods.
    
    When raised, it will print info about the function that raised it.
    """
    def __init__(self, msg: Optional[str] = None):
        msg = msg if msg else "Unimplemented method."
        msg = __info__(msg, 1)
        super().__init__(msg)
