from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional, TypeVar, Type
from fractions import Fraction

# conversions.py
# This file provides a framework for defining and registering unit conversions.
# It also provides a set of predefined conversions for common units.
# The syntax involved with calling the conversions can ensure readability and consistency with unit conversions.

FACTOR_CONVERSIONS = {}
"""
FACTOR_CONVERSIONS: Dict[str, Dict[str, float]]

- Allows conversions between units of measurement to be accessed via a dictionary.
- Only viable for ratio/factor conversions, not for more complex conversions.
- `FACTOR_CONVERSIONS[unit1][unit2]` will return the conversion factor from `unit1` to `unit2` if it exists.
- `register_factor(unit1:str, unit2:str, factor:float)` can be used to register a bidirectional conversion factor.
"""

FactorType = Tuple[str, str, float]

def register_factor(unit1:str, unit2:str, factor:float):
    """
    Register a conversion factor between two units of measurement. Registers bidirectionally
    
    Args:
        unit1 (str): The first unit of measurement.
        unit2 (str): The second unit of measurement.
        factor (float): The conversion factor from `unit1` to `unit2`.
    """
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
"""
METRIC_PREFIXES: Dict[str, int]

- A dictionary that maps metric prefixes to their respective powers of 10.
- Allows for easy recognition and conversion of metric units.
"""
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
"""
METRIC_PREFIX_LIST: List[str]

- A list of metric prefixes, sorted from smallest to largest.
"""
def setup_prefix_conversion(
    baseunit:str, 
    formatstr:str = "{prefix}{baseunit}", 
    prefixrange:List[str] = None, 
    factor_func:Callable[[float], float]=lambda x: x
):
    """
    Register conversion factors between a base unit and metric prefixes. 
    - formatstr allows for custom formatting of the unit names.
    - prefixrange allows for a subset of metric prefixes to be registered.
    - factor_func allows for automatic conversion of !1 dimensional units, such as m2, m3, etc.
    
    Args:
        baseunit (str): The base unit of measurement.
        formatstr (str): The format string for the unit names. Defaults to "{prefix}{baseunit}".
        prefixrange (List[str]): A list of metric prefixes to register. Defaults to all prefixes.
        factor_func (Callable[[float], float]): A function to apply to the metric prefix power of 10. Defaults to the identity function.
    """
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
        
def generic_network_conversion_setup(
    given_factors: Dict[str, Dict[str, float]]
)->List[FactorType]:
    """
    Given a dictionary of conversion edges in a unit system network, return a list of conversion factors to register.
    
    (FactorType = Tuple[str, str, float])
    
    Args:
        given_factors (Dict[str, Dict[str, float]]): A dictionary of conversion edges in a unit system network.
        
    Returns:
        result_factors (List[FactorType]): A list of conversion factors to register.
        
    Example:
    ```
    factors = {
        "a" : {"b" : 2, "c" : 3}, # Will trigger an exception if a->b->c is not equal to a->c
        "b" : {"c" : 4},
        "c" : {"d" : 5}
    }
    ```
    """
    given_factors_fraction: Dict[str, Dict[str, Fraction]] = {
        node1 : {node2 : Fraction(factor) for node2, factor in given_factors[node1].items()} for node1 in given_factors
    }
    result_factors: List[FactorType] = []
    edges: Set[Tuple[str, str]] = set()
    unique_key = lambda x: tuple(sorted(x))
    adjacency_matrix: Dict[str, Dict[str, Fraction]] = {}
    nodes: Dict[str, Set[str]] = {}
    all_nodes: Set[str] = set()
    
    def add_adjacency(node1:str, node2:str, factor:Fraction):
        nonlocal adjacency_matrix, edges, unique_key, result_factors, nodes
        adjacency_matrix[node1] = adjacency_matrix.get(node1, {})
        adjacency_matrix[node2] = adjacency_matrix.get(node2, {})
        def check_conflict(oldfactor, newfactor, node1, node2):
            if oldfactor != newfactor:
                raise ValueError(f"Conflicting factors for {node1} -> {node2}: {oldfactor} vs {newfactor}.")
        if node2 in adjacency_matrix[node1]:
            check_conflict(adjacency_matrix[node1][node2], factor, node1, node2)
        else:
            adjacency_matrix[node1][node2] = factor
        if node1 in adjacency_matrix[node2]:
            check_conflict(adjacency_matrix[node2][node1], 1 / factor, node2, node1)
        else:
            adjacency_matrix[node2][node1] = 1 / factor
        key = unique_key((node1, node2))
        if key not in edges:
            edges.add(key)
            nodes[node1].add(node2)
            nodes[node2].add(node1)
            if factor.numerator > factor.denominator:
                factor_float = float(factor.numerator / factor.denominator)
                result_factors.append((node2, node1, factor_float))
            else:
                factor_float = float(factor.denominator / factor.numerator)
                result_factors.append((node1, node2, factor_float))
    
    # quickly get all nodes
    for node1 in given_factors_fraction:
        all_nodes.add(node1)
        for node2 in given_factors_fraction[node1]:
            all_nodes.add(node2)
    # set up nodes
    for node in all_nodes:
        nodes[node] = set()
    # set up edges
    for node1 in given_factors_fraction:
        for node2 in given_factors_fraction[node1]:
            add_adjacency(node1, node2, given_factors_fraction[node1][node2])
    # set up transitive edges
    def transitive_propagate(start_node:str, current_node:str, visited:Optional[Set[str]] = None):
        nonlocal adjacency_matrix, nodes, add_adjacency
        if visited is None:
            visited = set()
        if current_node in visited or start_node == current_node:
            return
        visited.add(current_node)
        base_factor = adjacency_matrix[start_node][current_node]
        for node in nodes[current_node]:
            if node == start_node:
                continue
            factor = base_factor * adjacency_matrix[current_node][node]
            add_adjacency(start_node, node, factor)
            transitive_propagate(start_node, node, visited)
    for node in all_nodes:
        for subnode in list(nodes[node]):
            transitive_propagate(node, subnode)
    num_edges_per_node = {node : len(nodes[node]) for node in nodes}
    if not all([num_edges_per_node[node] == len(all_nodes) - 1 for node in all_nodes]):
        raise ValueError(f"Not all nodes are connected. {num_edges_per_node}\n{adjacency_matrix}")
    return result_factors

def setup_time_conversion():
    given_factors = {
        "seconds" : {"minutes" : 60},
        "minutes" : {"hours" : 60},
        "hours" : {"days" : 24},
        "days" : {"weeks" : 7}
    }
    factors = generic_network_conversion_setup(given_factors)
    for unit1, unit2, factor in factors:
        register_factor(unit1, unit2, factor)
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