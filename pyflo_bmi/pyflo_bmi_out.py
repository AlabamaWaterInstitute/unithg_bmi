from bmipy import Bmi
from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional
# from types import NoneType
import sys, os, re, warnings, enum
from pathlib import Path
import numpy as np

from numpy import ndarray
from .src import debug_utils as du
from .src.debug_utils import UnimplementedError
from .unit_hydrograph_model import unit_hydrograph_model

info_cats = {}
# category decorator
def _info_category(categories: List[str]):
    def decorator(func):
        global info_cats
        info_cats[func.__name__] = (func, categories)
        return func
    return decorator
# finalize the class decorator
def _finalize_info_categories(cls):
    global info_cats
    for name, (func, categories) in info_cats.items():
        _categories = []
        for category in categories:
            if category in cls._info_shorthands:
                _categories.extend(cls._info_shorthands[category])
            else:
                _categories.append(category)
        cls._info_cats[name] = _categories
        func.categories = _categories
    return cls

@_finalize_info_categories
class Bmi_Pyflo(Bmi):
    """
    A Basic Model Interface (BMI) wrapper for the PyFLO model.

    Required model methods:
    - Bookkeeping:
        - `initialize`
        - `update`
        - `update_until`
        - `finalize`
    - Time:
        - `get_start_time`
        - `get_end_time`
        - `get_current_time`
        - `get_time_step`
        - `get_time_units`
    - Variable Setters:
        - `set_value`
        - `set_value_at_indices`
    - Variable Getters:
        - `get_value`
        - `get_value_at_indices`
        - `get_value_ptr`
    - Variable Information:
        - `get_var_type`
        - `get_var_units`
        - `get_var_nbytes`
        - `get_var_itemsize`
        - `get_var_location`
        - `get_input_var_names`
        - `get_output_var_names`
        - `get_input_item_count`
        - `get_output_item_count`
    - Grid Information:
        - `get_grid_rank`
        - `get_grid_size`
        - `get_grid_type`
        - `get_grid_shape`
        - `get_grid_spacing`
        - `get_grid_origin`
        - `get_grid_x`
        - `get_grid_y`
        - `get_grid_z`
    """
    
    class VarType(enum.Enum):
        """
        Enumeration of variable types.
        """
        INPUT = 0
        OUTPUT = 1
        MODEL = 2
    
    class Var:
        """
        All 6 attributes compressed into a single class.
        Saves space and time over handling the 6 attributes separately.
        """
        _fields = ["name", "type", "value", "units", "nbytes", "itemsize", "location", "grid"]
        _defaults = ["", "float", 0.0, "m/s", 8, 8, "node", "none"]
        name, type, value, units, nbytes, itemsize, location, grid = _defaults
        value: np.ndarray
        def __init__(self, name: str, type: Optional[str] = None, value: Optional[Union[float, np.ndarray]] = None, units: Optional[str] = None, nbytes: Optional[int] = None, itemsize: Optional[int] = None, location: Optional[str] = None, grid: Optional[str] = None):
            """
            Initialize a new variable object.
            """
            self.name = name
            if value is not None:
                self.value = value if isinstance(value, np.ndarray) else np.array([value])
            for i, field in enumerate(self._fields):
                if hasattr(self, field):
                    # we already set this attribute
                    continue
                setattr(self, field, locals()[field] if locals()[field] is not None else self._defaults[i])
            if not isinstance(self.value, np.ndarray):
                self.value = np.array([self.value])
        def get_ptr(self):
            """
            Get a pointer to the value of the variable.
            """
            return self.value
        def get_value(self):
            """
            Get the value of the variable.
            """
            return self.value[0]
        def set_value(self, value: Union[float, np.ndarray]):
            """
            Set the value of the variable.
            """
            if isinstance(value, np.ndarray):
                self.value[0] = value[0]
            else:
                self.value[0] = value
    # Internal model attributes
    _name: str = "PyFlo_BMI"
    _start_time: float
    _num_time_steps: int
    _time_step_size: float
    _end_time: float
    _time: float
    _time_step: int
    _time_units: str = "s"
    # initialize in __init__ as instance variables
    _vars: dict[VarType, List[Var]] 
    _all_vars: dict[str, Var]
    _model_data: dict[str, Any]
    _model: Optional[object] = None
    
    _info_categories: List[str] = [
        "python-builtin",
        "getters",
        "setters",
        "timing",
        "bmi",
        "init",
        "finalize",
        "update",
        "variables",
        "grid",
        "values",
        "internal",
        "helpers",
        "no-calc",
    ]
    _info_shorthands: Dict[str, List[str]] = {
        "py-init": ["python-builtin", "init"],
        "py-finalize": ["python-builtin", "finalize"],
        "var-get": ["getters", "variables"],
        "var-set": ["setters", "variables"],
        "grid-get": ["getters", "grid"],
        "grid-set": ["setters", "grid"],
        "val-get": ["getters", "values"],
        "val-set": ["setters", "values"],
        "time-get": ["getters", "timing"],
        "bmi-init": ["bmi", "init"],
        "bmi-finalize": ["bmi", "finalize"],
    }
    _info_cats: Dict[str, List[str]] = {}
    
    suppress_info_categories: List[str] = [
        "no-calc",
        "init",
        "finalize",
        "bmi",
        "variables",
        "grid",
    ]
    
    track_variables: List[str]
        
    @_info_category(["py-init"])
    def __init__(self):
        """
        Initialize the model instance.
        """
        super(self.__class__, self).__init__()
        self.info()
        #Ensure these are initialized as instance variables, not class variables
        self._vars = {
            Bmi_Pyflo.VarType.INPUT: [],
            Bmi_Pyflo.VarType.OUTPUT: [],
            Bmi_Pyflo.VarType.MODEL: [],
        }
        self._model = None
        self._model_data = {}
        self._all_vars = {}
        self.track_variables = []
        
        self._time = 0.0
        self._time_step = 0
        self._time_step_size = 1.0
        self._num_time_steps = 720
        self._start_time = 0.0

    @_info_category(["internal", "helpers", "variables"])
    def _add_var(self, var:Var, vartype:VarType = VarType.MODEL) -> None:
        """
        Add a variable to the model.
        """
        self._all_vars[var.name] = var
        self._vars[vartype].append(var)

    @_info_category(["internal", "helpers", "variables"])
    def _add_vars(self, vars:List[Var], var_type:VarType = VarType.MODEL) -> None:
        """
        Add a list of variables to the model.
        """
        self.info()
        for var in vars:
            self._add_var(var, var_type)
    

    def info(self, msg:str = None) -> None:
        call_func = du.__func__(1)
        if call_func in self._info_cats:
            categories = self._info_cats[call_func]
            if any(cat in self.suppress_info_categories for cat in categories):
                return
        send_msg = f"'{__name__}';"
        send_msg += du.__info__(msg, offset=1)
        print(send_msg, file=sys.stderr, flush=True)



    # BMI: Bookkeeping

    @_info_category(["bmi-init"])
    def initialize(self, config_file: str = None) -> None:
        """
        Initialize the model.
        """
        self.info()
        self._start_time = self.get_start_time()
        config_vars = []
        if config_file:
            config_vars = self.read_config(config_file)
        if len(config_vars) == 0:
            config_vars = [
                self.Var("area_sqkm", "float", 0.0, "km^2", 8, 8, "node", "none")
            ]
            self.track_variables.append("area_sqkm")
            # self.track_variables.append("discharge_calculated")
        vars_for_input = []
        forcing_vars = [
                self.Var("DLWRF_surface", "float", 0.0, "W/m^2", 8, 8, "node", "none"),
                self.Var("PRES_surface", "float", 0.0, "Pa", 8, 8, "node", "none"),
                self.Var("SPFH_2maboveground", "float", 0.0, "kg/kg", 8, 8, "node", "none"),
                self.Var("precip_rate", "float", 0.0, "kg/m^2/s", 8, 8, "node", "none"),
                self.Var("DSWRF_surface", "float", 0.0, "W/m^2", 8, 8, "node", "none"),
                self.Var("TMP_2maboveground", "float", 0.0, "K", 8, 8, "node", "none"),
                self.Var("UGRD_10maboveground", "float", 0.0, "m/s", 8, 8, "node", "none"),
                self.Var("VGRD_10maboveground", "float", 0.0, "m/s", 8, 8, "node", "none"),
                self.Var("APCP_surface", "float", 0.0, "kg/m^2", 8, 8, "node", "none"),
            ]
        vars_for_input.extend(forcing_vars)
        self._add_vars(
            vars_for_input,
            var_type=self.VarType.INPUT
        )
        
        self._add_vars(
            config_vars,
            var_type=self.VarType.MODEL
        )


        # Output variables
        self._add_var(
            self.Var("discharge_calculated", "float", 0.0, "m^3/s", 8, 8, "node", "none"),
            vartype=self.VarType.OUTPUT
        )
        
    # read_config
    @_info_category(["bmi", "init"])
    def read_config(self, config_file: str) -> List[Var]:
        """
        Read the model configuration from a file.
        """
        self.info()
        print(f"Reading config file '{config_file}'...", file=sys.stderr, flush=True)
        if not isinstance(config_file, str):
            raise ValueError("Config file must be a string.")
        _config_file = Path(config_file).resolve()
        if not _config_file.exists():
            raise FileNotFoundError(f"Config file '{config_file}' not found.")
        with open(_config_file, "r") as f:
            config = f.read()
        if config == "":
            return []
        if config is None:
            raise ValueError("Config file is empty.")
        # Parse the config
        config_lines = config.split("\n")
        type_coercion = {
            "float": float,
            "double": float,
            "int": int,
            "str": str,
            "bool": bool,
        }
        header = config_lines[0]
        header = header.split(",")
        header = [h.strip() for h in header]
        acceptable_headers = [
            "name",
            "type",
            "value",
            "units",
            "nbytes",
            "itemsize",
            "location",
            "grid",
        ]
        for h in header:
            if h not in acceptable_headers:
                raise ValueError(f"Invalid header '{h}' in config file.")
        config_vars = []
        for line in config_lines[1:]:
            if len(line) == 0:
                continue
            line = line.split(",")
            line = [l.strip() for l in line]
            if len(line) != len(header):
                raise ValueError(f"Invalid line '{line}' in config file. Expected {len(header)} columns, got {len(line)}.")
            var = {}
            for i, h in enumerate(header):
                var[h] = line[i]
            if var["type"] in type_coercion:
                var["value"] = type_coercion[var["type"]](var["value"])
            var["nbytes"] = int(var["nbytes"])
            var["itemsize"] = int(var["itemsize"])
            _var = self.Var(**var)
            config_vars.append(_var)
            self.info(f"Added config var '{_var.name}' with value '{_var.value}'.")
        return config_vars
                

    @_info_category(["bmi", "update"])
    def update(self) -> None:
        """
        Update the model by one time step.
        """
        self.info()
        # # quick bypass for now
        # self._time += self._time_step_size
        # return
        if self._model is None:
            area_sqkm = float(self.get_value("area_sqkm"))
            self._model = unit_hydrograph_model(area=area_sqkm, interval=5)
        # send one hour of rainfall to the model
        rain_depth = float(self.get_value("APCP_surface"))
        # result = float(self._model.get_current_flow_period(60))
        self._model.add_rain_period(rain_depth, 60)
        # receive one hour of discharge from the model
        result = float(self._model.get_current_flow_period(60))
        # if result > 0:
        #     print(f"{self._time}]Rain depth: {rain_depth}, Discharge: {result}.", file=sys.stderr, flush=True)
        self.set_value("discharge_calculated", result)
        assert self.get_value("discharge_calculated") == result, f"Discharge set to {result}, but get_value returned {self.get_value('discharge_calculated')}."
        self._time += self._time_step_size
        self._time_step += 1

    @_info_category(["bmi", "update"])
    def update_until(self, time: float) -> None:
        """
        Update the model until the given time.
        """
        self.info()
        # for _ in range(int((time - self._time) / self._time_step_size)):
        #     self.update()
        self._time = time
        self.update()

    @_info_category(["bmi-finalize"])
    def finalize(self) -> None:
        """
        Finalize the model.
        """
        self.info()
        pass
    
    @_info_category(["bmi", "getters", "no-calc"])
    def get_component_name(self) -> str:
        """
        Get the model name.
        """
        self.info()
        return self._name

    # BMI: Time

    @_info_category(["bmi", "time-get", "no-calc"])
    def get_start_time(self) -> float:
        """
        Get the model start time.
        """
        self.info()
        return 0.0
    
    @_info_category(["bmi", "time-get", "no-calc"])
    def get_end_time(self) -> float:
        """
        Get the model end time.
        """
        self.info()
        return self.get_start_time() + self._time_step_size * self._num_time_steps
    
    @_info_category(["bmi", "time-get", "no-calc"])
    def get_current_time(self) -> float:
        """
        Get the model current time.
        """
        self.info()
        return self._time
    
    @_info_category(["bmi", "time-get", "no-calc"])
    def get_time_step(self) -> float:
        """
        Get the model time step.
        """
        self.info()
        return self._time_step
    
    @_info_category(["bmi", "time-get", "no-calc"])
    def get_time_units(self) -> str:
        """
        Get the model time units.
        """
        self.info()
        return self._time_units
    
    # BMI: Variable Setters

    @_info_category(["bmi", "var-get", "no-calc", "internal", "helpers"])
    def _get_var(self, name: str) -> Var:
        """
        Get a variable object by name.
        """
        self.info()
        return self._all_vars.get(name, None)
    
    @_info_category(["bmi", "var-set"])
    def set_value(self, name: str, value: Any) -> None:
        """
        Set the value of a variable.
        """
        self.info()
        if name in self.track_variables:
            print(f"Setting variable '{name}' to value '{value}'.", file=sys.stderr, flush=True)
        var = self._get_var(name)
        if var is not None:
            var.set_value(value)
        else:
            raise ValueError(f"Variable '{name}' not found.")
        
    @_info_category(["bmi", "var-set"])
    def set_value_at_indices(self, name: str, indices: np.ndarray, src: np.ndarray) -> None:
        """
        Set the value of a variable at specific indices.
        """
        self.info()
        var_ptr = self.get_value_ptr(name)
        var_ptr[indices] = src

    # BMI: Variable Getters

    @_info_category(["bmi", "var-get", "no-calc"])
    def get_value(self, name: str) -> float:
        """
        Get the value of a variable.
        """
        self.info()
        var = self._get_var(name)
        if var is not None:
            return var.get_value()
        else:
            raise ValueError(f"Variable '{name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_value_at_indices(self, name: str, dest: np.ndarray, inds: np.ndarray) -> np.ndarray:
        """
        Get the value of a variable at specific indices.
        """
        self.info()
        var_ptr = self.get_value_ptr(name)
        dest[inds] = var_ptr[inds]
        return dest

    @_info_category(["bmi", "var-get", "no-calc"])
    def get_value_ptr(self, name: str) -> Any:
        """
        Get a pointer to the value of a variable.
        """
        self.info()
        var = self._get_var(name)
        if var is not None:
            return var.get_ptr()
        else:
            raise ValueError(f"Variable '{name}' not found.")

    # BMI: Variable Information

    @_info_category(["bmi", "var-get", "no-calc"])
    def get_var_type(self, var_name: str) -> str:
        """
        Get the type of a variable.
        """
        self.info()
        var = self._get_var(var_name)
        if var is not None:
            return var.type
        else:
            raise ValueError(f"Variable '{var_name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_var_units(self, var_name: str) -> str:
        """
        Get the units of a variable.
        """
        self.info()
        var = self._get_var(var_name)
        if var is not None:
            return var.units
        else:
            raise ValueError(f"Variable '{var_name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_var_nbytes(self, var_name: str) -> int:
        """
        Get the number of bytes of a variable.
        """
        self.info()
        var = self._get_var(var_name)
        if var is not None:
            return var.nbytes
        else:
            raise ValueError(f"Variable '{var_name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_var_itemsize(self, var_name: str) -> int:
        """
        Get the item size of a variable.
        """
        self.info()
        var = self._get_var(var_name)
        if var is not None:
            return var.itemsize
        else:
            raise ValueError(f"Variable '{var_name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_var_location(self, var_name: str) -> str:
        """
        Get the location of a variable.
        """
        self.info()
        var = self._get_var(var_name)
        if var is not None:
            return var.location
        else:
            raise ValueError(f"Variable '{var_name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_var_grid(self, name: str) -> int:
        """
        Get the grid of a variable.
        """
        self.info()
        var = self._get_var(name)
        if var is not None:
            return var.grid
        else:
            raise ValueError(f"Variable '{name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_input_var_names(self) -> List[str]:
        """
        Get the names of the input variables.
        """
        self.info()
        return [var.name for var in self._vars[self.VarType.INPUT]]
    
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_output_var_names(self) -> List[str]:
        """
        Get the names of the output variables.
        """
        self.info()
        return [var.name for var in self._vars[self.VarType.OUTPUT]]
    
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_input_item_count(self) -> int:
        """
        Get the number of input items.
        """
        self.info()
        return len(self._vars[self.VarType.INPUT])
    
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_output_item_count(self) -> int:
        """
        Get the number of output items.
        """
        self.info()
        return len(self._vars[self.VarType.OUTPUT])
    
    # BMI: Grid Information

    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_edge_count(self, grid: int) -> int:
        """
        Get the number of edges in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_edge_nodes(self, grid: int, edge_nodes: ndarray) -> ndarray:
        """
        Get the nodes of an edge in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_face_count(self, grid: int) -> int:
        """
        Get the number of faces in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_face_edges(self, grid: int, face_edges: ndarray) -> ndarray:
        """
        Get the edges of a face in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_face_nodes(self, grid: int, face_nodes: ndarray) -> ndarray:
        """
        Get the nodes of a face in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_node_count(self, grid: int) -> int:
        """
        Get the number of nodes in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_nodes_per_face(self, grid: int, nodes_per_face: ndarray) -> ndarray:
        """
        Get the number of nodes per face in a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_origin(self, grid: int, origin: ndarray) -> ndarray:
        """
        Get the origin of a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_rank(self, grid: int) -> int:
        """
        Get the rank of a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_shape(self, grid: int, shape: ndarray) -> ndarray:
        """
        Get the shape of a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_size(self, grid: int) -> int:
        """
        Get the size of a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_spacing(self, grid: int, spacing: ndarray) -> ndarray:
        """
        Get the spacing of a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_type(self, grid: int) -> str:
        """
        Get the type of a grid.
        """
        self.info()
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_x(self, grid: int, x: ndarray) -> ndarray:
        """
        Get the x-coordinates of a grid.
        """
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_y(self, grid: int, y: ndarray) -> ndarray:
        """
        Get the y-coordinates of a grid.
        """
        raise UnimplementedError()
    
    @_info_category(["bmi", "grid-get", "no-calc"])
    def get_grid_z(self, grid: int, z: ndarray) -> ndarray:
        """
        Get the z-coordinates of a grid.
        """
        raise UnimplementedError()
        
