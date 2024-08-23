from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional
import numpy as np
import sys, os
# from matplotlib import pyplot as plt
if __name__ == "__main__":
    from src.discrete_time_series import Discretime
else:
    from .src.discrete_time_series import Discretime

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
    units = ["seconds", "minutes", "hours", "days", "weeks"]
    factors = {
        "minutes" : {"seconds" : 60},
        "hours" : {"minutes" : 60},
        "days" : {"hours" : 24},
        "weeks" : {"days" : 7}
        }
    paths = []
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
        
UUID = 0
def get_uuid()->int:
    global UUID
    UUID += 1
    return UUID

def readable_ndarray(arr:np.ndarray)->str:
    lines = []
    for i in range(arr.shape[0]):
        line = []
        for j in range(arr.shape[1]):
            line.append(f"{arr[i, j]:.2f}")
        lines.append("[" + ", ".join(line) + "]")
    return "[" + ",\n".join(lines) + "]"

def get_shape(obj:Any)->Tuple[int, int]:
    if isinstance(obj, np.ndarray):
        return obj.shape
    if hasattr(obj, "shape"):
        return obj.shape
    if hasattr(obj, "__len__"):
        if hasattr(obj, "__getitem__"):
            firstitem = obj[0]
            if hasattr(firstitem, "__len__"):
                return (len(obj), len(firstitem))
        return (len(obj),)
    if isinstance(obj, (int, float)):
        return "Scalar"
    return (0,)

class Basin:
    """
    Absolute bare minimum basin class
    
    Attributes:
        area (float): The delineated region concentrating to a point, in square kilometers.
        ordinates (numpy.ndarray): The unscaled unit hydrograph runoff distribution. Time steps in minutes.
        ordinate_geom_type (Literal["right", "midpoint", "left", "trapezoid"]): The type of geomorphological ordinate.
    """
    area:float
    ordinates:Discretime
    ordinate_geom_type:Literal["right", "midpoint", "left", "trapezoid"]
    def __init__(self, area:float, ordinates:Union[np.ndarray, Discretime], ordinate_geom_type:Literal["right", "midpoint", "left", "trapezoid"]="right"):
        """
        Initialize the basin.
        
        Args:
            area (float): The delineated region concentrating to a point, in square kilometers (`areasqkm|area_sqkm`).
            ordinates (Union[np.ndarray, Discretime]): The unscaled unit hydrograph runoff distribution in `Minutes` and `Unitless`.
            ordinate_geom_type (Literal["right", "midpoint", "left", "trapezoid"]): The type of geomorphological ordinate.
        """
        self.area = area
        if isinstance(ordinates, np.ndarray):
            self.ordinates = Discretime(ordinates, time_unit="minutes", data_unit="unitless")
        else:
            self.ordinates = ordinates
        self.ordinate_geom_type = ordinate_geom_type
        
    def volume_to_depth(self, volume: Union[float, np.ndarray, Discretime])->Union[float, np.ndarray, Discretime]:
        """
        Convert volume to depth.
        
        Args:
            volume (float): The volume to convert, in cubic meters.
            
        Returns:
            float: The depth, in millimeters.
        """
        area_m2 = self.area * FACTOR_CONVERSIONS["sqkm"]["sqm"]
        depth_m = volume / area_m2
        depth_mm = depth_m * FACTOR_CONVERSIONS["m"]["mm"]
        return depth_mm
    
    def depth_to_volume(self, depth: Union[float, np.ndarray, Discretime])->Union[float, np.ndarray, Discretime]:
        """
        Convert depth to volume.
        
        Args:
            depth (float): The depth to convert, in millimeters.
            
        Returns:
            float: The volume, in cubic meters.
        """
        if isinstance(depth, Discretime):
            if depth.data_unit != "mm":
                raise ValueError(f"Discretime data unit must be millimeters: {depth.data_unit}")
            depth.data_unit = "m3"
        # depth_m = depth / FACTOR_CONVERSIONS["m"]["mm"]
        # area_m2 = self.area * FACTOR_CONVERSIONS["sqkm"]["sqm"]
        # volume = depth_m * area_m2
        conversion_factor = self.area * FACTOR_CONVERSIONS["sqkm"]["sqm"] * FACTOR_CONVERSIONS["mm"]["m"]
        depth = depth * conversion_factor
        return depth
    
    @staticmethod
    def interpolate(array: Union[np.ndarray, Discretime], interval: int) -> np.ndarray:
        """
        Interpolate the array to a new interval.
        
        Args:
            array (Union[np.ndarray, Discretime]): The array to interpolate.
            interval (int): The interval to interpolate to, in minutes.
            
        Returns:
            np.ndarray: The interpolated array.
        """
        if isinstance(array, Discretime):
            if array.interval == interval:
                return array.data.copy()
        return Discretime.discrete_interpolate(array, interval)
    
    def segment_area_under_curve(self, curve:np.ndarray, index:int)->float:
        """
        Calculate the area under a segment of the curve.
        
        Args:
            curve (numpy.ndarray): The curve to calculate the area under.
            index (int): The index of the segment to calculate the area under.
        
        Returns:
            float: The area under the segment of the curve.
        """
        if self.ordinate_geom_type == "right":
            x1, y1 = curve[index]
            x2, y2 = curve[index+1]
            return (x2 - x1) * y1
        elif self.ordinate_geom_type == "left":
            x1, y1 = curve[index]
            x2, y2 = curve[index+1]
            return (x2 - x1) * y2
        elif self.ordinate_geom_type == "midpoint":
            x1, y1 = curve[index]
            x2, y2 = curve[index+1]
            return (x2 - x1) * (y1 + y2) / 2
        elif self.ordinate_geom_type == "trapezoid":
            x1, y1 = curve[index]
            x2, y2 = curve[index+1]
            return (x2 - x1) * (y1 + y2) / 2
        else:
            raise ValueError("Invalid ordinate geom type")
        
    def discrete_segment_area(self, curve:Discretime, index:int)->float:
        """
        Alternative method for segment_area_under_curve.
        Since the Discretime class enforces evenly-spaced data, 
        we can avoid working with x values until the end.

        Args:
            curve (Discretime): The curve to calculate the area under.
            index (int): The index of the segment to calculate the area under.
            
        Returns:
            float: The area under the segment of the curve.
                The ordinate is assumed to be unitless, so the area is in minutes..?
        """
        if self.ordinate_geom_type == "right":
            return curve[index] * curve.interval
        elif self.ordinate_geom_type == "left":
            return curve[index+1] * curve.interval
        elif self.ordinate_geom_type == "midpoint":
            return (curve[index] + curve.data[index+1, 1]) * curve.interval / 2
        elif self.ordinate_geom_type == "trapezoid":
            return (curve[index] + curve.data[index+1, 1]) * curve.interval / 2
        else:
            raise ValueError("Invalid ordinate geom type")
            
    def ordinate_area_under_curve(self, ordinate:np.ndarray)->float:
        """
        Calculate the area under the curve of the ordinate.
        
        Args:
            ordinate (numpy.ndarray): The ordinate to calculate the area under the curve for.
            
        Returns:
            float: The area under the curve.
        """
        if isinstance(ordinate, Discretime):
            ordinate = ordinate.data
        area = 0
        for i in range(ordinate.shape[0]-1):
            area += self.segment_area_under_curve(ordinate, i)
        return area
    
    def discrete_ordinate_area(self, ordinate:Discretime)->float:
        """
        Alternative method for ordinate_area_under_curve.
        Since the Discretime class enforces evenly-spaced data, 
        we can avoid working with x values until the end.

        Args:
            ordinate (Discretime): The ordinate to calculate the area under the curve for.
            
        Returns:
            float: The area under the curve. 
                The ordinate is assumed to be unitless, so the area is in minutes..?
        """
        area = 0
        for i in range(ordinate.data.shape[0]-1):
            area += self.discrete_segment_area(ordinate, i)
        return area
        
    def unit_hydrograph(self, interval:int)->np.ndarray:
        """
        Generate a unit hydrograph for the basin.
        
        Args:
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            numpy.ndarray: The generated unit hydrograph.
        """
        interpolated = self.interpolate(self.ordinates, interval)
        # print(type(interpolated))
        area = self.ordinate_area_under_curve(interpolated)
        interpolated[:, 1] /= area
        return interpolated
    
    def discrete_unit_hydrograph(self, interval:int)->Discretime:
        """
        Generate a unit hydrograph for the basin.
        
        Args:
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            Discretime: The generated unit hydrograph.
        """
        if interval == self.ordinates.interval:
            interpolated = self.ordinates.copy()
        else:
            ordinates = Discretime.discrete_interpolate(self.ordinates.data, interval)
            interpolated = Discretime(ordinates, time_unit="minutes", data_unit="unitless", interval=interval)
        area = self.discrete_ordinate_area(interpolated)
        interpolated /= area
        return interpolated
    
    def flood_data(self, rain_depths:np.ndarray, interval:int)->np.ndarray:
        """
        Generate pairs of basin runoff flow generated from rainfall over time.
        
        Args:
            rain_depths (numpy.ndarray): A 2D array of scaled rainfall depths over time.
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            numpy.ndarray: The generated pairs of time and runoff flow generated from rainfall.
        """
        hydrograph = self.unit_hydrograph(interval)
        rain_depths = rain_depths.copy()
        # rain_depths = self.interpolate(rain_depths, interval)
        rain_depths[:, 1] = self.depth_to_volume(rain_depths[:, 1])
        latest_time = np.max(rain_depths[:, 0])
        hydrograph_length = np.max(hydrograph[:, 0])
        longest_time = int(latest_time + hydrograph_length)
        flood_data = np.zeros((longest_time // interval + 1, 2))
        flood_data[:, 0] = np.arange(0, longest_time + interval, interval)
        for i in range(rain_depths.shape[0]):
            time, depth = rain_depths[i]
            for j in range(hydrograph.shape[0] - 1):
                area_under_curve = self.segment_area_under_curve(hydrograph, j)
                time_offset = hydrograph[j, 0]
                timestep = int(time + time_offset) // interval
                flood_data[timestep, 1] += area_under_curve * depth
        return flood_data
    
    def discrete_flood_data(self, rain_depths:Discretime, interval:int)->Discretime:
        """
        Generate pairs of basin runoff flow generated from rainfall over time.
        
        Args:
            rain_depths (Discretime): A 2D array of scaled rainfall depths over time in minutes and millimeters.
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            Discretime: The generated pairs of time and runoff flow generated from rainfall,
                in minutes and cubic meters.
        """
        hydrograph = self.discrete_unit_hydrograph(interval)
        rain_depths = rain_depths.copy()
        rain_depths = self.depth_to_volume(rain_depths)
        flood_data = rain_depths.copy() * 0
        for i in range(rain_depths.data.shape[0]):
            time, depth = rain_depths.data[i]
            flood_curve = hydrograph.offset(time)
            flood_curve *= depth * interval
            flood_data += flood_curve
        return flood_data
    
    def discrete_flood_data_incomplete(self, rain_depths:Discretime, interval:int, start_ind:int, end_ind:int)->Discretime:
        """
        Calculate and return basin runoff flow over a subset of time.
        
        Args:
            rain_depths (Discretime): A 2D array of scaled rainfall depths over time in minutes and millimeters.
            interval (int): The interval to increment calculations by, in minutes.
            start_ind (int): The index to start calculating from.
            end_ind (int): The index to stop calculating at.
            
        Returns:
            Discretime: The generated pairs of time and runoff flow generated from rainfall,
                in minutes and cubic meters.
        """
        hydrograph = self.discrete_unit_hydrograph(interval)
        # contamination_period: how long before the time segment to start calculating
        contamination_period = hydrograph.end_index - hydrograph.start_index
        rain_depths = rain_depths.copy()
        rain_depths = self.depth_to_volume(rain_depths)
        output_data = np.zeros((end_ind - start_ind, 2))
        output_data[:, 0] = rain_depths.data[start_ind:end_ind, 0]
        output_data = Discretime(output_data, time_unit="minutes", data_unit="m3", interval=interval)
        prestart = start_ind - contamination_period
        prestart = max(rain_depths.start_index, prestart)
        # print(output_data.data.shape)
        for i in range(prestart, start_ind):
            time, depth = rain_depths.data[i]
            flood_curve = hydrograph.offset(time)
            flood_curve *= depth * interval
            output_data += flood_curve
        for i in range(start_ind, end_ind):
            time, depth = rain_depths.data[i]
            flood_curve = hydrograph.offset(time)
            flood_curve *= depth * interval
            output_data += flood_curve
        return Discretime(output_data[:end_ind-start_ind, :], time_unit="minutes", data_unit="m3", interval=interval)
        

    
    def flood_data_partials(self, rain_depths: np.ndarray, interval: int)->List[np.ndarray]:
        """
        Generate a list of flood curves created by individual rainfall events.
        """
        if isinstance(rain_depths, Discretime):
            return self.discrete_flood_data_partials(rain_depths, interval)
        hydrograph = self.unit_hydrograph(interval)
        rain_depths = rain_depths.copy()
        rain_depths[:, 1] = self.depth_to_volume(rain_depths[:, 1])
        latest_time = np.max(rain_depths[:, 0])
        hydrograph_length = np.max(hydrograph[:, 0])
        longest_time = int(latest_time + hydrograph_length)
        flood_data = np.zeros((longest_time // interval + 1, 2))
        flood_data[:, 0] = np.arange(0, longest_time + interval, interval)
        num_rain_events = np.count_nonzero(rain_depths[:, 1])
        partials = [flood_data.copy() for i in range(num_rain_events)]
        n=0
        for i in range(rain_depths.shape[0]):
            time, depth = rain_depths[i]
            if depth == 0:
                continue
            for j in range(hydrograph.shape[0] - 1):
                area_under_curve = self.segment_area_under_curve(hydrograph, j)
                time_offset = hydrograph[j, 0]
                timestep = int(time + time_offset) // interval
                partials[n][timestep, 1] += area_under_curve * depth
            n+=1
        return partials
    
    def discrete_flood_data_partials(self, rain_depths: Discretime, interval: int)->List[Discretime]:
        """
        Generate a list of flood curves created by individual rainfall events.
        """
        hydrograph = self.discrete_unit_hydrograph(interval)
        rain_depths = rain_depths.copy()
        rain_depths.apply(self.depth_to_volume)
        flood_data = rain_depths.copy() * 0
        num_rain_events = np.count_nonzero(rain_depths.data[:, 1])
        partials = [flood_data.copy() for i in range(num_rain_events)]
        n=0
        for i in range(rain_depths.data.shape[0]):
            time, depth = rain_depths.data[i]
            if depth == 0:
                continue
            flood_curve = hydrograph.offset(time)
            flood_curve *= depth * interval
            partials[n] += flood_curve
            n+=1
        return partials
    
    
def cumulative_sum(array:np.ndarray)->np.ndarray:
    """
    Calculate the cumulative sum array for the input 2D array.
    
    Args:
        array (numpy.ndarray): The 2D input array. Time, Value pairs.
        
    Returns:
        numpy.ndarray: The cumulative 2D array. Time, Cumulative Value pairs.
    """
    if isinstance(array, Discretime):
        array = array.data
    cumsum = np.zeros(array.shape)
    cumsum[0, 0] = array[0, 0]
    cumsum[0, 1] = array[0, 1]
    for i in range(1, array.shape[0]):
        cumsum[i, 0] = array[i, 0]
        cumsum[i, 1] = cumsum[i-1, 1] + array[i, 1]
    return cumsum

class unit_hydrograph_model:
    default_curve:np.ndarray = np.array([
        [0.0, 0.0], [0.1, 0.022456], [0.2, 0.074853], [0.3, 0.142221], [0.4, 0.232045],
        [0.5, 0.35181], [0.6, 0.49403], [0.7, 0.613795], [0.8, 0.696134], [0.9, 0.741046],
        [1.0, 0.748531], [1.1, 0.741046], [1.2, 0.696134], [1.3, 0.643737], [1.4, 0.583854],
        [1.5, 0.509001], [1.6, 0.419177], [1.7, 0.344324], [1.8, 0.291927], [1.9, 0.247015],
        [2.0, 0.209589], [2.1, 0.182267], [2.2, 0.154946], [2.3, 0.13249], [2.4, 0.110034],
        [2.5, 0.095063], [2.6, 0.080093], [2.7, 0.068865], [2.8, 0.057637], [2.9, 0.049403],
        [3.0, 0.041169], [3.1, 0.035555], [3.2, 0.029941], [3.3, 0.025824], [3.4, 0.021707],
        [3.5, 0.018713], [3.6, 0.015719], [3.7, 0.013474], [3.8, 0.011228], [3.9, 0.009731],
        [4.0, 0.008234], [4.1, 0.007336], [4.2, 0.006437], [4.3, 0.005539], [4.4, 0.004641],
        [4.5, 0.003743], [4.6, 0.002994], [4.7, 0.002246], [4.8, 0.001497], [4.9, 0.000749],
        [5.0, 0.0]
    ])
    area: float
    interval:float
    basin:Basin
    input_rain:Discretime
    timestep:int
    calc_step:int
    _flow_cache:Optional[np.ndarray]
    uuid:int
    def __init__(self, area:float, interval:float, ordinates:Optional[np.ndarray]=None, ordinate_geom_type:Literal["right", "midpoint", "left", "trapezoid"]="right"):
        """
        Initialize the unit hydrograph model.
        
        Args:
            area (float): The area of the basin, in square kilometers.
            interval (float): The interval to increment calculations by, in minutes.
            ordinates (numpy.ndarray): The unscaled unit hydrograph runoff distribution in minutes and unitless.
            ordinate_geom_type (Literal["right", "midpoint", "left", "trapezoid"]): The type of geomorphological ordinate.
        """
        self.area = area
        self.interval = interval
        if ordinates is None:
            ordinates = self.default_curve.copy()
            ordinates[:, 0] *= 60
        self.basin = Basin(area, ordinates, ordinate_geom_type)
        array_size = 256
        flow_array = np.zeros((array_size, 2))
        flow_array[:, 0] = np.arange(array_size) * interval
        self.input_rain = Discretime(flow_array, time_unit="minutes", data_unit="mm")
        self.timestep = 0
        self.calc_step = -1
        self._flow_cache = None
        self.uuid = get_uuid()
        
    def add_rain(self, depth: float):
        """
        Add a rain to the model.
        
        Args:
            depth (float): The depth of the rain, in millimeters.
        """
        self.timestep += 1
        if self.timestep >= self.input_rain.data.shape[0]:
            self.input_rain.resize()
        self.input_rain[self.timestep, 1] += depth
        
    def add_rain_period(self, depth: float, period: int):
        """
        Add a period of rain to the model.
        
        Args:
            depth (float): The depth of the rain, in millimeters.
            period (int): The period of time to apply the rain, in minutes.
        """
        # self.timestep += 1
        start_time = self.timestep
        rain_period_width = period // self.interval
        if self.timestep + rain_period_width >= self.input_rain.data.shape[0]:
            self.input_rain.resize()
        self.input_rain[self.timestep:self.timestep+rain_period_width, 1] += depth
        self.timestep += rain_period_width
        end_time = self.timestep
        # if depth > 0:
        #     print(f"[{self.uuid}] Added rain of {depth}mm over {period} minutes and {rain_period_width} intervals from {start_time} to {end_time}")
        
    @property
    def flows(self)->Discretime:
        """
        Get the flows.
        
        When not cached, the `input_rain` (in `mm`) is given to the `basin` to generate the flows.
        
        The `basin` returns the flows in `cubic meters`.
        
        Returns:
            Discretime: The flows in cubic meters per interval.
        """
        if self.calc_step == self.timestep:
            return self._flow_cache
        self._flow_cache = self.basin.discrete_flood_data(self.input_rain, self.interval)
        self.calc_step = self.timestep
        # sum_flows = np.sum(self._flow_cache.data[:self.timestep, 1])
        # if sum_flows > 0:
        #     print(f"[{self.uuid}] Calculated flows for timestep {self.timestep}")
        return self._flow_cache
    
    def get_flows(self, step_start: Optional[int], step_end: Optional[int])->Discretime:
        """
        Get the flows.
        
        Returns:
            Discretime: The flows in cubic meters per interval.
        """
        if step_start is None:
            step_start = 0
        if step_end is None:
            step_end = self.timestep
        return self.flows[step_start:step_end]
        
    def get_current_flow(self, timestep:int=None)->float:
        """
        Get the current flow.
        
        Args:
            timestep (int): The timestep to get the flow for.
            
        Returns:
            float: The flow at the timestep in cubic meters per interval.
        """
        if timestep is None:
            timestep = self.timestep
        return self.flows[timestep]
    
    def get_current_flow_period(self, period:int, end:Optional[int]=None)->float:
        """
        Get the flow over a period.
        
        Args:
            period (int): The period to get the flow over, in minutes.
            end (int): The end timestep to get the flow to.
            
        Returns:
            float: The flow over the period in cubic meters per interval.
        """
        if end is None:
            end = self.timestep + 1
        flow_period_width = period // self.interval
        flow_vals = self.basin.discrete_flood_data_incomplete(self.input_rain, self.interval, end-flow_period_width, end)
        flow_amt = np.sum(flow_vals.data[:, 1])
        # if flow_amt > 0:
        #     print(f"[{self.uuid}] Calculated flow {flow_amt} over period {period} from {end-flow_period_width} to {end}")
        return flow_amt
    
if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from pathlib import Path
    thisdir = Path(__file__).parent.parent
    dist_dir = thisdir / "dist"
    basindisplay = dist_dir / "basin_display"
    basindisplay.mkdir(exist_ok=True)
    # ordinates = np.array([[0, 0], [10, 1], [20, 0]])
    ordinates = np.array([
        [0.0, 0.0], [0.1, 0.03], [0.2, 0.1], [0.3, 0.19], [0.4, 0.31], [0.5, 0.47], [0.6, 0.66], 
        [0.7, 0.82], [0.8, 0.93], [0.9, 0.99], [1.0, 1.0], [1.1, 0.99], [1.2, 0.93], [1.3, 0.86], 
        [1.4, 0.78], [1.5, 0.68], [1.6, 0.56], [1.7, 0.46], [1.8, 0.39], [1.9, 0.33], [2.0, 0.28], 
        [2.2, 0.207], [2.4, 0.147], [2.6, 0.107], [2.8, 0.077], [3.0, 0.055], [3.2, 0.04], 
        [3.4, 0.029], [3.6, 0.021], [3.8, 0.015], [4.0, 0.011], [4.5, 0.005], [5.0, 0.0], 
    ])
    ordinates[:, 0] *= 60
    ordinates = Discretime.discrete_interpolate(ordinates, interval=5)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(ordinates[:,0], ordinates[:,1])
    ax.scatter(ordinates[:,0], ordinates[:,1])
    ax.set_title("Initial Hydrograph Ordinates")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Runoff Value")
    fig.savefig(basindisplay / "initial_ordinates.png")
    
    
    basin = Basin(1, ordinates)
    hydrograph_5 = basin.discrete_unit_hydrograph(5)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(hydrograph_5[:,0], hydrograph_5[:,1])
    ax.scatter(hydrograph_5[:,0], hydrograph_5[:,1])
    ax.set_title("Unit Hydrograph")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Runoff (normalized)")
    fig.savefig(basindisplay / "unit_hydrograph.png")
    
    raindata = np.zeros((30, 2))
    raindata[:,0] = np.arange(30) * 5
    raindata[1:20,1] = 0.1
    raindata = Discretime(raindata, time_unit="minutes", data_unit="mm")
    
    flooddata = basin.discrete_flood_data(raindata, 5)
    fig, ax = plt.subplots()
    ax:plt.Axes
    raindata_m3 = raindata.copy()
    raindata_m3[:,1] = basin.depth_to_volume(raindata[:,1])
    ax.plot(flooddata[:,0], flooddata[:,1], color="red")
    # ax.scatter(flooddata[:,0], flooddata[:,1], color="red")
    ax.plot(raindata_m3[:,0], raindata_m3[:,1], color="blue")
    # ax.scatter(raindata[:,0], raindata[:,1], color="blue")
    ax.set_title("Flood Data")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Runoff")
    fig.savefig(basindisplay / "flood_data.png")
    
    flooddata_partials = basin.flood_data_partials(raindata, 5)
    fig, ax = plt.subplots()
    ax:plt.Axes
    # barax = ax.twinx()
    for i, partial in enumerate(flooddata_partials):
        lineplot = ax.plot(partial[:,0], partial[:,1], label=f"Event {i}")
        # barax.bar(partial[:,0], partial[:,1], color=lineplot[0].get_color(), alpha=0.2, width=5)
    ybound0, ybound1 = ax.get_ybound()
    # barax.set_ybound(ybound0 * 1.1, ybound1 * 1.1)
    ax.set_ybound(ybound0 * 1.1, ybound1 * 1.1)
    # barax.invert_yaxis()
    ax.plot(raindata_m3[:,0], raindata_m3[:,1], color="blue", label="Rainfall")
    ax.plot(flooddata[:,0], flooddata[:,1], color="red", label="Total", linestyle="--")
    ax.set_title("Flood Data Partials")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Runoff")
    ax.legend()
    fig.savefig(basindisplay / "flood_data_partials.png")
    
    cumulative_rainfall_m3 = cumulative_sum(raindata_m3)
    cumulative_flood = cumulative_sum(flooddata)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(cumulative_rainfall_m3[:,0], cumulative_rainfall_m3[:,1], color="blue", label="Rainfall")
    ax.plot(cumulative_flood[:,0], cumulative_flood[:,1], color="red", label="Total")
    ax.set_title("Cumulative Rainfall and Flood")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Runoff")
    ax.legend()
    fig.savefig(basindisplay / "cumulative_rainfall_flood.png")
    
    
    # Test the unit hydrograph model
    uhmodel = unit_hydrograph_model(1, 5)
    recv = []
    recv_time = []
    for _time, _depth in raindata:
        uhmodel.add_rain(_depth)
        recv.append(uhmodel.get_current_flow())
        recv_time.append(_time)
    flows = uhmodel.flows
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_xlabel(f"Time ({raindata.time_unit})")
    ax.set_ylabel(f"Flow ({flows.data_unit})")
    ax.set_title("Unit Hydrograph Model Flows")
    ax.plot(recv_time, recv, label="Model Flows", color="red")
    ax.plot(flooddata[:,0], flooddata[:,1], label="Prev Flows", color="blue", linestyle="--")
    ax.plot(raindata_m3[:,0], raindata_m3[:,1], label="Rainfall", color="green", linestyle="--")
    ax.legend()
    fig.savefig(basindisplay / "uhmodel_flows.png")
    
    # Test the unit hydrograph model with hour-long periods like nextgen uses.
    hour_num = 24
    rain_hours = 7
    hour_raindata = np.zeros((hour_num, 2))
    hour_raindata[:,0] = np.arange(hour_num) * 60
    hour_raindata[1:rain_hours+1:2,1] = 0.1
    hour_raindata = Discretime(hour_raindata, time_unit="minutes", data_unit="mm")
    uhmodel = unit_hydrograph_model(1, 5, ordinates)
    recv = []
    recv_time = []
    for _time, _depth in hour_raindata:
        uhmodel.add_rain_period(_depth, 60)
        recv.append(uhmodel.get_current_flow_period(60))
        recv_time.append(_time)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_xlabel(f"Time ({hour_raindata.time_unit})")
    ax.set_ylabel(f"Flow ({flows.data_unit})")
    ax.set_title("Unit Hydrograph Model Flows")
    ax.plot(recv_time, recv, label="Model Flows", color="red")
    fig.savefig(basindisplay / "uhmodel_recv_flows_hourly.png")
    flows = uhmodel.flows
    real_raindepths = []
    real_raintimes = []
    extra_raintime = 60//5
    for _time, _depth in hour_raindata:
        real_raindepths.extend([_depth] * extra_raintime)
        real_raintimes.extend([_time + i for i in range(0, 60, 5)])
    real_raindata = np.array([real_raintimes, real_raindepths]).T
    real_raindata = Discretime(real_raindata, time_unit="minutes", data_unit="mm")
    real_raindata_m3 = real_raindata.copy()
    real_raindata_m3 = basin.depth_to_volume(real_raindata_m3)
    graphed_hour_raindata = basin.depth_to_volume(hour_raindata)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_xlabel(f"Time ({hour_raindata.time_unit})")
    ax.set_ylabel(f"Flow ({flows.data_unit})")
    ax.set_title("Unit Hydrograph Model Flows")
    # ax.plot(recv_time, recv, label="Recieved Flows", color="red")
    ax.plot(flows.times, flows.values, label="Model Flows", color="blue")
    ax.plot(real_raindata_m3.times, real_raindata_m3.values, label="Rainfall", color="green")
    for i in range(0, hour_num):
        if i in range(1, rain_hours + 1, 2):
            ax.axvline(i*60, color="black", linestyle="--", alpha=0.5)
            ax.text(i*60, graphed_hour_raindata[i], f"hour {i}", rotation=-40, verticalalignment="center")
        elif i==0 or (i > rain_hours and i % 2 == 1):
            ax.axvline(i*60, color="black", linestyle="--", alpha=0.5)
    ax.legend()
    fig.savefig(basindisplay / "uhmodel_flows_hourly.png")
    
    # cumulative hour rainfall and flood
    cumulative_hour_rainfall_m3 = cumulative_sum(real_raindata_m3)
    cumulative_hour_flood = cumulative_sum(flows)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(cumulative_hour_rainfall_m3[:,0], cumulative_hour_rainfall_m3[:,1], color="blue", label="Rainfall")
    ax.plot(cumulative_hour_flood[:,0], cumulative_hour_flood[:,1], color="red", label="Total")
    ax.set_title("Cumulative Rainfall and Flood")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Runoff")
    ax.legend()
    fig.savefig(basindisplay / "cumulative_hour_rainfall_flood.png")
    
    
    
    
    