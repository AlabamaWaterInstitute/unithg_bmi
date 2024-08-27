from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional
import numpy as np


IntervalType = Union[float, int]
class Discretime:
    """ 
    Object for handling discrete time series data.
    Wraps a numpy array, with additional functionality.
    
    Attributes:
        data (numpy.ndarray): The time series data. Default is an empty array.
        interval (float): The fixed interval between data points. Default is 1.
        time_unit (str): The unit of time for the data. Default is "minutes".
        data_unit (str): The unit of the data. Default is "unitless".
    """
    data:np.ndarray
    values:np.ndarray
    times:np.ndarray
    interval:IntervalType
    time_unit:Literal["seconds", "minutes", "hours", "days", "weeks"]
    data_unit:str
    def __init__(self, data:np.ndarray=np.zeros(shape=(1, 2)), interval:IntervalType=-1, time_unit:Literal["seconds", "minutes", "hours", "days", "weeks"]="minutes", data_unit:str="unitless"):
        self.data = data
        self.check_ndarray(self.data)
        self.interval = interval if interval > 0 else np.max(np.diff(self.data[:, 0]))
        self.data = self.discretize(self.data, self.interval)
        self.time_unit = time_unit
        self.data_unit = data_unit
        
    @property
    def times(self)->np.ndarray:
        return self.data[:, 0]
    
    @property
    def values(self)->np.ndarray:
        return self.data[:, 1]
    
    @property
    def end_time(self)->float:
        return np.max(self.times)
    
    @property
    def start_time(self)->float:
        return np.min(self.times)
    
    @property
    def start_index(self)->int:
        return Discretime.round_to_interval(self.start_time, self.interval)
    
    @property
    def end_index(self)->int:
        return Discretime.round_to_interval(self.end_time, self.interval)
    
    @property
    def duration(self)->float:
        return self.end_time - self.start_time
    
    def __repr__(self)->str:
        return repr(self.data)
    
    @staticmethod
    def round_to_interval(time:float, interval:IntervalType)->int:
        """
        Round a time to the nearest interval.
        
        Args:
            time (float): The time to round.
            interval (Union[float, int]): The interval to round to.
            
        Returns:
            int: The index of the rounded time relative to the origin.
        """
        return int(np.round(time / interval))
    
    @staticmethod
    def discretize(data: np.ndarray, interval:IntervalType)->np.ndarray:
        """
        Make a time series data array discrete.
        
        Args:
            data (numpy.ndarray): The data to make discrete.
            interval (Union[float, int]): The interval to round to.
            
        Returns:
            numpy.ndarray: The discrete data array.
        """
        min_time, max_time = np.min(data[:, 0]), np.max(data[:, 0])
        mindex = Discretime.round_to_interval(min_time, interval)
        maxdex = Discretime.round_to_interval(max_time, interval)
        new_data = np.zeros((maxdex - mindex + 1, 2))
        new_data[:, 0] = np.arange(mindex, maxdex + 1) * interval
        for i in range(data.shape[0]):
            time = data[i, 0]
            value = data[i, 1]
            index = Discretime.round_to_interval(time, interval) - mindex
            new_data[index, 1] += value
        return new_data
    
    @staticmethod
    def discrete_interpolate(data:np.ndarray, interval:IntervalType)->np.ndarray:
        """
        Interpolate a discrete time series data array.
        
        Args:
            data (numpy.ndarray): The data to interpolate.
            interval (Union[float, int]): The interval to interpolate to.
            
        Returns:
            numpy.ndarray: The interpolated data array.
        """
        # from matplotlib import pyplot as plt, lines
        # fig, axs = plt.subplots(4)
        # axs:List[plt.Axes]
        # axs[0].plot(data[:, 0], data[:, 1], label="Original", color="red")
        # axs[0].scatter(data[:, 0], data[:, 1], color="red")
        # axs[0].set_title("Original Data")
        if isinstance(data, Discretime):
            data = data.data
        # print(data.shape)
        # Enforce that the data is discrete at the given interval
        discrete = Discretime.discretize(data, interval)
        # print(discrete.shape)
        # axs[1].plot(discrete[:, 0], discrete[:, 1], label="Discrete", color="blue")
        # axs[1].scatter(discrete[:, 0], discrete[:, 1], color="blue")
        # axs[1].set_title("Discrete Data")
        occupied_indices = {i: False for i in range(discrete.shape[0])}
        for i in range(data.shape[0]):
            time = data[i, 0]
            index = Discretime.round_to_interval(time, interval)
            occupied_indices[index] = True
        # print(occupied_indices)
        last_nonzero = -1
        for i in range(discrete.shape[0]):
            if occupied_indices[i] and last_nonzero == -1:
                last_nonzero = i
            elif occupied_indices[i]:
                start = last_nonzero
                end = i
                if end - start == 1:
                    # print(f"Skipping {last_nonzero} to {end}")
                    last_nonzero = i
                    continue
                start_val = discrete[start, 1]
                end_val = discrete[end, 1]
                # start_time = discrete[start, 0]
                # end_time = discrete[end, 0]
                # line_arr = [
                #     lines.Line2D([start_time, start_time], [0, start_val], color="red", linestyle="--"),
                #     lines.Line2D([end_time, end_time], [0, end_val], color="red", linestyle="--"),
                #     lines.Line2D([start_time, end_time], [start_val, end_val], color="red", linestyle="--"),
                # ]
                # print(f"Adding line from {start_time} to {end_time} with values {start_val} to {end_val}")
                # for line in line_arr:
                #     axs[2].add_line(line)
                # axs[2].set_title("Interpolation")
                slope = (end_val - start_val) / (end - start)
                for j in range(start + 1, end):
                    discrete[j, 1] = start_val + slope * (j - start)
                last_nonzero = i
            # else:
            #     print(f"{last_nonzero} to {i} is zero")
        # axs[2].autoscale()
        # axs[3].plot(discrete[:, 0], discrete[:, 1], label="Interpolated", color="green")
        # axs[3].plot(data[:, 0], data[:, 1], label="Original", color="blue")
        # axs[3].scatter(discrete[:, 0], discrete[:, 1], color="green")
        # axs[3].scatter(data[:, 0], data[:, 1], color="blue")
        # axs[3].set_title("Interpolated Data")
        # plt.show()
        
        return discrete
    
    def adjust_interval(self, new_interval:IntervalType)->"Discretime":
        """
        Adjust the interval of the data.
        
        Args:
            new_interval (Union[float, int]): The new interval to adjust to.
            
        Returns:
            Discretime: The adjusted Discretime object.
        """
        new_data = Discretime.discretize(self.data, new_interval)
        return Discretime(new_data, new_interval, self.time_unit, self.data_unit)
    
    def match_units(self, other:"Discretime")->None:
        if self.time_unit != other.time_unit:
            raise ValueError(f"Time units must match: {self.time_unit} != {other.time_unit}")
        if self.data_unit == "unitless" or other.data_unit == "unitless":
            return
        if self.data_unit != other.data_unit:
            raise ValueError(f"Data units must match: {self.data_unit} != {other.data_unit}")
        
    @staticmethod
    def check_ndarray(data:np.ndarray)->None:
        row, col = data.shape
        if row < 1:
            raise ValueError(f"Data must have at least one row: {data.shape}")
        if col != 2:
            raise ValueError(f"Data must have two columns: {data.shape}")
        
    
    @staticmethod
    def discrete_add(data1:np.ndarray, data2:np.ndarray, interval:IntervalType)->np.ndarray:
        """
        Add two discrete time series data arrays.
        
        Args:
            data1 (numpy.ndarray): The first data array.
            data2 (numpy.ndarray): The second data array.
            interval (Union[float, int]): The allowed interval between data points.
            
        Returns:
            numpy.ndarray: The sum of the two data arrays.
        """
        disc1, disc2 = data1, data2
        if isinstance(data1, Discretime):
            if data1.interval != interval:
                disc1 = data1.adjust_interval(interval)
        elif isinstance(data1, np.ndarray):
            Discretime.check_ndarray(data1)
            disc1 = Discretime(data1, interval)
        else:
            raise ValueError(f"Unsupported type for addition: {type(data1)}")
        if isinstance(data2, Discretime):
            if data2.interval != interval:
                disc2 = data2.adjust_interval(interval)
        elif isinstance(data2, np.ndarray):
            Discretime.check_ndarray(data2)
            disc2 = Discretime(data2, interval)
        else:
            raise ValueError(f"Unsupported type for addition: {type(data2)}")
        start1, end1 = disc1.start_index, disc1.end_index
        start2, end2 = disc2.start_index, disc2.end_index
        mindex = min(start1, start2)
        maxdex = max(end1, end2)
        new_data = np.zeros((maxdex - mindex + 1, 2))
        new_data[:, 0] = np.arange(mindex, maxdex + 1) * interval
        # print(f"Adding data from {start1} to {end1} and {start2} to {end2}")
        # print(f"New data shape: {new_data.shape} from {disc1.data.shape} and {disc2.data.shape}")
        start1, start2 = start1 - mindex, start2 - mindex
        end1, end2 = end1 - mindex, end2 - mindex
        new_data[start1:end1+1, 1] += disc1.values
        new_data[start2:end2+1, 1] += disc2.values
        return new_data
    
    def copy(self)->"Discretime":
        """
        Create a copy of the Discretime object.
        
        Returns:
            Discretime: The copied Discretime object.
        """
        return Discretime(self.data.copy(), self.interval, self.time_unit, self.data_unit)
    
    def __iadd__(self, other)->"Discretime":
        if isinstance(other, Discretime):
            self.match_units(other)
            self.data = Discretime.discrete_add(self, other, self.interval)
        elif isinstance(other, np.ndarray):
            Discretime.check_ndarray(other)
            self.data = Discretime.discrete_add(self, other, self.interval)
        elif isinstance(other, (int, float)):
            self.data[:, 1] += other
        else:
            raise ValueError(f"Unsupported type for addition: {type(other)}")
        return self
    
    def __add__(self, other)->"Discretime":
        new = self.copy()
        new += other
        return new
    
    def __isub__(self, other)->"Discretime":
        if isinstance(other, Discretime):
            self.match_units(other)
            self.data = Discretime.discrete_add(self.data, -other.data, self.interval)
        elif isinstance(other, np.ndarray):
            Discretime.check_ndarray(other)
            self.data = Discretime.discrete_add(self.data, -other, self.interval)
        elif isinstance(other, (int, float)):
            self.data[:, 1] -= other
        else:
            raise ValueError(f"Unsupported type for subtraction: {type(other)}")
        return self
    
    def __sub__(self, other)->"Discretime":
        new = self.copy()
        new -= other
        return new
    
    def __imul__(self, other)->"Discretime":
        # Implemented only for scalar multiplication
        if isinstance(other, (int, float)):
            self.data[:, 1] *= other
        else:
            raise ValueError(f"Unsupported type for multiplication: {type(other)}")
        return self
    
    def __mul__(self, other)->"Discretime":
        new = self.copy()
        new *= other
        return new
    
    def __itruediv__(self, other)->"Discretime":
        # Implemented only for scalar division
        if isinstance(other, (int, float)):
            self.data[:, 1] /= other
        else:
            raise ValueError(f"Unsupported type for division: {type(other)}")
        return self
    
    def __truediv__(self, other)->"Discretime":
        new = self.copy()
        new /= other
        return new
    
    def __ifloordiv__(self, other)->"Discretime":
        # Implemented only for scalar division
        if isinstance(other, (int, float)):
            self.data[:, 1] //= other
        else:
            raise ValueError(f"Unsupported type for division: {type(other)}")
        return self
    
    def __floordiv__(self, other)->"Discretime":
        new = self.copy()
        new //= other
        return new
    
    def __getitem__(self, index:Union[int, slice, tuple[Any, Any]])->Union[float, np.ndarray]:
        if isinstance(index, int):
            return self.data[index, 1]
        elif isinstance(index, slice):
            return self.data[index, 1]
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(f"Unsupported index: {index}")
            return self.data[index]
        else:
            raise ValueError(f"Unsupported index: {index}")
        
    def __setitem__(self, index:Union[int, slice, tuple[Any, Any]], value:Union[float, np.ndarray])->None:
        if isinstance(index, int):
            self.data[index, 1] = value
        elif isinstance(index, slice):
            self.data[index, 1] = value
        elif isinstance(index, tuple):
            if len(index) != 2:
                raise ValueError(f"Unsupported index: {index}")
            self.data[index] = value
        else:
            raise ValueError(f"Unsupported index: {index}")
        
    def apply(self, func:Callable[[float], float])->"Discretime":
        """
        Apply a function to the data values.
        
        Args:
            func (Callable[[float], float]): The function to apply.
        """
        self.data[:, 1] = func(self.data[:, 1])
        return self
    
    def offset(self, offset:float)->"Discretime":
        """
        Return a version of the data with an offset applied.
        
        Args:
            offset (float): The offset to apply.
            
        Returns:
            Discretime: The offset Discretime object.
        """
        new_data = self.data.copy()
        new_data[:, 0] += offset
        return Discretime(new_data, self.interval, self.time_unit, self.data_unit)
    
    def resize(self, new_size:Optional[int] = None)->None:
        """
        Resize the data array.
        
        Args:
            new_size (Optional[int]): The new size of the data array. Default is double the current size.
        """
        if new_size is None:
            new_size = self.data.shape[0] * 2
        old_start = self.start_time
        old_data = self.data
        self.data = np.zeros((new_size, 2))
        self.data[:, 0] = np.arange(new_size) * self.interval + old_start
        self.data[:old_data.shape[0], :] = old_data
        # print(f"Resized data from {old_data.shape[0]} to {new_size}")
        
    def __iter__(self)->np.ndarray:
        return iter(self.data)
    
    def add_at_time(self, time:float, value:float)->None:
        """
        Add a value at a specific time.
        
        Args:
            time (float): The time to add the value.
            value (float): The value to add.
        """
        index = Discretime.round_to_interval(time, self.interval)
        if index >= self.data.shape[0]:
            self.resize(index + 1)
        self.data[index, 1] += value
        
    def add_at_period(self, time_start:float, time_end:float, value:float)->None:
        """
        Add a value over a specific time period.
        
        Args:
            time_start (float): The start time of the period.
            time_end (float): The end time of the period.
            value (float): The value to add.
        """
        start_index = Discretime.round_to_interval(time_start, self.interval)
        end_index = Discretime.round_to_interval(time_end, self.interval)
        if end_index >= self.data.shape[0]:
            self.resize(end_index + 1)
        self.data[start_index:end_index+1, 1] += value
        
    def get_data(self)->np.ndarray:
        """
        Get the section of the data array that contains data.
        
        Returns:
            numpy.ndarray: The data array.
        """
        _start = self.start_index
        _end = self.end_index
        for i in range(_start, _end + 1):
            if self.data[i, 1] != 0:
                _start = i
                break
        for i in range(_end, _start - 1, -1):
            if self.data[i, 1] != 0:
                _end = i
                break
        return self.data[_start:_end+1]
    
    
if __name__ == "__main__":
    initial_hydrograph = np.array([
        [0.0, 0.0], [0.1, 0.03], [0.2, 0.1], [0.3, 0.19], [0.4, 0.31], [0.5, 0.47], [0.6, 0.66], 
        [0.7, 0.82], [0.8, 0.93], [0.9, 0.99], [1.0, 1.0], [1.1, 0.99], [1.2, 0.93], [1.3, 0.86], 
        [1.4, 0.78], [1.5, 0.68], [1.6, 0.56], [1.7, 0.46], [1.8, 0.39], [1.9, 0.33], [2.0, 0.28], 
        [2.2, 0.207], [2.4, 0.147], [2.6, 0.107], [2.8, 0.077], [3.0, 0.055], [3.2, 0.04], 
        [3.4, 0.029], [3.6, 0.021], [3.8, 0.015], [4.0, 0.011], [4.5, 0.005], [5.0, 0.0], 
    ])
    initial_hydrograph[:, 0] *= 60
    # initial_hydrograph = Discretime(initial_hydrograph, 0.1)
    Discretime.discrete_interpolate(initial_hydrograph, 5)