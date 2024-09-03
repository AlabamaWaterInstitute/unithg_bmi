from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional, TypeVar

FACTOR_CONVERSIONS = {}
def register_factor(unit1:str, unit2:str, factor:float):
    global FACTOR_CONVERSIONS
    if unit1 not in FACTOR_CONVERSIONS:
        FACTOR_CONVERSIONS[unit1] = {}
    FACTOR_CONVERSIONS[unit1][unit2] = factor
    if unit2 not in FACTOR_CONVERSIONS:
        FACTOR_CONVERSIONS[unit2] = {}
    FACTOR_CONVERSIONS[unit2][unit1] = 1 / factor
register_factor("in", "mm", 25.4)
register_factor("sqkm", "acres", 247.1)
register_factor("sqkm", "sqm", 1e6)
METRIC_PREFIXES = {}
# prefixes for small units, separated by 1e3
_regular_prefix_short = ["y", "z", "a", "f", "p", "n", "u", "m", "", "k", "M", "G", "T", "P", "E", "Z", "Y"]
METRIC_PREFIXES.update({
    f"{prefix}" : -24 + 3*i for i, prefix in enumerate(_regular_prefix_short)
})
METRIC_PREFIXES.update({
    "c" : -2, "d" : -1, "da" : 1, "h" : 2
})
METRIC_PREFIXES = dict(sorted(METRIC_PREFIXES.items(), key=lambda x: x[1]))
METRIC_PREFIX_LIST = list(METRIC_PREFIXES.keys())
def setup_prefix_conversion(baseunit:str, formatstr:str="{prefix}{baseunit}", prefixrange:List[str]=None, factor_func:Callable[[float], float]=lambda x: x):
    global METRIC_PREFIXES
    metric_keys = list(METRIC_PREFIXES.keys())
    if prefixrange:
        metric_keys = [key for key in metric_keys if key in prefixrange]
    if "" not in metric_keys:
        metric_keys.append("")
    # print(metric_keys)
    for i in range(len(metric_keys)):
        key1 = metric_keys[i]
        unit1 = formatstr.format(prefix=key1, baseunit=baseunit)
        factor1 = factor_func(METRIC_PREFIXES[key1])
        for j in range(len(metric_keys)):
            key2 = metric_keys[j]
            unit2 = formatstr.format(prefix=key2, baseunit=baseunit)
            factor2 = factor_func(METRIC_PREFIXES[key2])
            factor_exp = factor1 - factor2
            factor = 10 ** factor_exp
            register_factor(unit1, unit2, factor)
setup_prefix_conversion("m")
setup_prefix_conversion("s", prefixrange=METRIC_PREFIX_LIST[0:METRIC_PREFIX_LIST.index("m") + 1])
setup_prefix_conversion("m3", factor_func=lambda x: 3*x)
setup_prefix_conversion("m2", factor_func=lambda x: 2*x)
register_factor("m2", "sqm", 1)
register_factor("km2", "sqkm", 1)
def setup_time_conversion():
    units: List[str] = ["seconds", "minutes", "hours", "days", "weeks"]
    factors: Dict[str, Dict[str, int]] = {
        "minutes" : {"seconds" : 60},
        "hours" : {"minutes" : 60},
        "days" : {"hours" : 24},
        "weeks" : {"days" : 7}
        }
    paths: List[List[str]] = [
        ["minutes", "seconds"],
        ["hours", "minutes"],
        ["days", "hours"],
        ["weeks", "days"]
    ]
    def calc_path_factor(path:List[str]):
        factor = 1
        for i in range(len(path)-1):
            unit1 = path[i]
            unit2 = path[i+1]
            factor *= factors[unit1][unit2]
        return factor
    def find_path(unit1:str, unit2:str, path:List[str]=[]):
        if unit1 == unit2:
            start = path[0]
            end = path[-1]
            if start not in factors:
                factors[start] = {}
            if end not in factors[start]:
                factors[start][end] = calc_path_factor(path)
                paths.append(path)
            elif len(paths) == 0:
                paths.append(path)
            return True
        if unit1 not in factors:
            path.remove(unit1)
            if find_path(unit2, unit1, [unit2]):
                return True
        for unit in factors[unit1]:
            if unit in path:
                continue
            if find_path(unit, unit2, path + [unit]):
                return True
    for unit1 in units:
        for unit2 in units:
            if unit1 == unit2:
                continue
            find_path(unit1, unit2, [unit1])
            # print(paths[-1], unit1, unit2)
    for path in paths:
        factor = 1
        for i in range(len(path)-1):
            unit1 = path[i]
            unit2 = path[i+1]
            factor *= factors[unit1][unit2]
        # print(path, factor)
        register_factor(path[0], path[-1], factor)
                
setup_time_conversion()

if __name__ == "__main__":
    def test_conversion(unit1: str, unit2: str, expected: Union[float, int]):
        factor = FACTOR_CONVERSIONS[unit1][unit2]
        assert factor == expected, f"Expected factor {expected} for conversion from {unit1} to {unit2}, got {factor}"
    def test_basic_conversions():
        test_conversion("in", "mm", 25.4)
        test_conversion("mm", "in", 1/25.4)
        test_conversion("sqkm", "acres", 247.1)
        test_conversion("acres", "sqkm", 1/247.1)
        test_conversion("sqkm", "sqm", 1e6)
        test_conversion("sqm", "sqkm", 1e-6)
    def test_metric_prefix_conversions():
        test_conversion("m", "km", 1e-3)
        test_conversion("km", "m", 1e3)
        test_conversion("m", "mm", 1e3)
        test_conversion("mm", "m", 1e-3)
        test_conversion("m3", "km3", 1e-9)
        test_conversion("km3", "m3", 1e9)
        test_conversion("m2", "km2", 1e-6)
        test_conversion("km2", "m2", 1e6)
        test_conversion("m2", "sqm", 1)
        test_conversion("sqm", "m2", 1)
        test_conversion("km2", "sqkm", 1)
        test_conversion("sqkm", "km2", 1)
    def test_time_conversions():
        test_conversion("seconds", "minutes", 1/60)
        test_conversion("minutes", "seconds", 60)
        test_conversion("minutes", "hours", 1/60)
        test_conversion("hours", "minutes", 60)
        test_conversion("hours", "days", 1/24)
        test_conversion("days", "hours", 24)
        test_conversion("days", "weeks", 1/7)
        test_conversion("weeks", "days", 7)
    test_basic_conversions()
    test_metric_prefix_conversions()
    test_time_conversions()