from bmipy import Bmi
from typing import List, Dict, Tuple, Any
import sys, os

from numpy import ndarray
from .src import debug_utils as du
from .src.debug_utils import UnimplementedError
from .pyflo_model import unit_hydrograph_model

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
    
    class Var:
        """
        All 6 attributes compressed into a single class.
        Saves space and time over handling the 6 attributes separately.
        """
        _fields = ["name", "type", "value", "units", "nbytes", "itemsize", "location", "grid"]
        _defaults = ["", "double", 0.0, "m/s", 8, 8, "node", "none"]
        def __init__(self, name: str, type: str = None, value: float = None, units: str = None, nbytes: int = None, itemsize: int = None, location: str = None, grid: str = None):
            """
            Initialize a new variable object.
            """
            self.name = name
            for i, field in enumerate(self._fields[1:]):
                setattr(self, field, locals()[field] if locals()[field] is not None else self._defaults[i])
    # Internal model attributes
    _name: str = "PyFlo_BMI"
    _start_time: float = 0.0
    _num_time_steps: int = 24
    _time_step_size: float = 1.0
    _end_time: float = 0.0
    _time: float = 0.0
    _time_step: int = 0
    _time_units: str = "s"
    _input_vars: list[Var] = []
    _output_vars: list[Var] = []
    _vars: dict[str, Var] = {}
    _model_data: dict[str, Any] = {}
    _model: object = None
    
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
        
    @_info_category(["py-init"])
    def __init__(self):
        """
        Initialize the model instance.
        """
        super(self.__class__, self).__init__()
        self.info()
        # Input variables
        ## Hydrograph-sourced variables
        # self._add_vars(
        #     [
        #         self.Var("areasqkm", "float", 0.0, "km^2", 8, 8, "node", "none"),
        #         self.Var("lengthkm", "float", 0.0, "km", 8, 8, "node", "none"),
        #         self.Var("tot_drainage_areasqkm", "float", 0.0, "km^2", 8, 8, "node", "none"),
        #     ],
        #     is_input=True
        # )
        ## Forcing variables
        # (time,DLWRF_surface,PRES_surface,SPFH_2maboveground,precip_rate,DSWRF_surface,TMP_2maboveground,UGRD_10maboveground,VGRD_10maboveground,APCP_surface)
        # skip time, it's a string of some format.
        
    @_info_category(["internal", "helpers", "variables"])
    def _add_var(self, var:Var, is_input:bool = False) -> None:
        """
        Add a variable to the model.
        """
        self._vars[var.name] = var
        if is_input:
            self._input_vars.append(var)
        else:
            self._output_vars.append(var)

    @_info_category(["internal", "helpers", "variables"])
    def _add_vars(self, vars:List[Var], is_input:bool = False) -> None:
        """
        Add a list of variables to the model.
        """
        self.info()
        for var in vars:
            self._add_var(var, is_input)

    @_info_category(["internal", "helpers", "variables", "bmi-init"])
    def _add_model_inputs(self) -> None:
        """
        Add the model input variables.
        """
        self.info()
        self._add_vars(
            [
                self.Var("areasqkm", "float", 0.0, "km^2", 8, 8, "node", "none"),
                self.Var("lengthkm", "float", 0.0, "km", 8, 8, "node", "none"),
                self.Var("tot_drainage_areasqkm", "float", 0.0, "km^2", 8, 8, "node", "none"),
            ],
            is_input=True
        )
        

    def info(self):
        call_func = du.__func__(1)
        if call_func in self._info_cats:
            categories = self._info_cats[call_func]
            if any(cat in self.suppress_info_categories for cat in categories):
                return
        msg = f"'{__name__}';"
        msg += du.__info__(msg="", offset=1)
        print(msg, file=sys.stderr, flush=True)



    # BMI: Bookkeeping

    @_info_category(["bmi-init"])
    def initialize(self, config_file: str = None) -> None:
        """
        Initialize the model.
        """
        self.info()
        self._start_time = self.get_start_time()
        self._add_model_inputs()
        self._add_vars(
            [
                self.Var("DLWRF_surface", "float", 0.0, "W/m^2", 8, 8, "node", "none"),
                self.Var("PRES_surface", "float", 0.0, "Pa", 8, 8, "node", "none"),
                self.Var("SPFH_2maboveground", "float", 0.0, "kg/kg", 8, 8, "node", "none"),
                self.Var("precip_rate", "float", 0.0, "kg/m^2/s", 8, 8, "node", "none"),
                self.Var("DSWRF_surface", "float", 0.0, "W/m^2", 8, 8, "node", "none"),
                self.Var("TMP_2maboveground", "float", 0.0, "K", 8, 8, "node", "none"),
                self.Var("UGRD_10maboveground", "float", 0.0, "m/s", 8, 8, "node", "none"),
                self.Var("VGRD_10maboveground", "float", 0.0, "m/s", 8, 8, "node", "none"),
                self.Var("APCP_surface", "float", 0.0, "kg/m^2", 8, 8, "node", "none"),
            ],
            is_input=True
        )


        # Output variables
        self._add_var(
            self.Var("discharge_calculated", "float", 0.0, "m^3/s", 8, 8, "node", "none"),
            is_input=False
        )

    @_info_category(["bmi", "update"])
    def update(self) -> None:
        """
        Update the model by one time step.
        """
        self.info()
        # quick bypass for now
        self._time += self._time_step_size
        return
        if self._model is None:
            areasqkm = self.get_value("areasqkm")
            self._model = unit_hydrograph_model(area=areasqkm, duration=self._num_time_steps * self._time_step_size, interval=self._time_step_size)
        result = self._model.step()
        self.set_value("discharge_calculated", result)
        self._time += self._time_step_size

    @_info_category(["bmi", "update"])
    def update_until(self, time: float) -> None:
        """
        Update the model until the given time.
        """
        self.info()
        for _ in range(int((time - self._time) / self._time_step_size)):
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
        if name in self._vars:
            return self._vars[name]
        for var in self._input_vars:
            if var.name == name:
                return var
        for var in self._output_vars:
            if var.name == name:
                return var
        return None
    
    @_info_category(["bmi", "var-set"])
    def set_value(self, name: str, value: float) -> None:
        """
        Set the value of a variable.
        """
        self.info()
        var = self._get_var(name)
        if var is not None:
            var.value = value
        else:
            raise ValueError(f"Variable '{name}' not found.")
        
    @_info_category(["bmi", "var-set"])
    def set_value_at_indices(self, name: str, indices: List[int], value: float) -> None:
        """
        Set the value of a variable at specific indices.
        """
        self.info()
        raise UnimplementedError()

    # BMI: Variable Getters

    @_info_category(["bmi", "var-get", "no-calc"])
    def get_value(self, name: str) -> float:
        """
        Get the value of a variable.
        """
        self.info()
        var = self._get_var(name)
        if var is not None:
            return var.value
        else:
            raise ValueError(f"Variable '{name}' not found.")
        
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_value_at_indices(self, name: str, indices: List[int]) -> float:
        """
        Get the value of a variable at specific indices.
        """
        self.info()
        raise UnimplementedError()

    @_info_category(["bmi", "var-get", "no-calc"])
    def get_value_ptr(self, name: str) -> Any:
        """
        Get a pointer to the value of a variable.
        """
        self.info()
        raise UnimplementedError()

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
        return [var.name for var in self._input_vars]
    
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_output_var_names(self) -> List[str]:
        """
        Get the names of the output variables.
        """
        self.info()
        return [var.name for var in self._output_vars]
    
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_input_item_count(self) -> int:
        """
        Get the number of input items.
        """
        self.info()
        return len(self._input_vars)
    
    @_info_category(["bmi", "var-get", "no-calc"])
    def get_output_item_count(self) -> int:
        """
        Get the number of output items.
        """
        self.info()
        return len(self._output_vars)
    
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
        
