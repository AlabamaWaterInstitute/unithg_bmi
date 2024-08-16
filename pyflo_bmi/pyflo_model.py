from typing import List, Dict, Tuple, Any
from pyflo import system, distributions
from pyflo.nrcs import hydrology
import numpy as np
import sys, os
# from matplotlib import pyplot as plt

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

class Basin(hydrology.Basin):
    area:float
    cn:float
    tc:float
    runoff_dist:np.ndarray
    peak_time:float
    peak_factor:float
    def unit_hydrograph(self, interval):
        """Get a hydrograph that represents the time-flow relationship per unit (inch) of depth.

        Args:
            interval (float): The amount of time the output will increment by.

        Returns:
            numpy.ndarray: The hydrograph of potential basin runoff.

        """
        runoff_dist = self.runoff_dist
        peak_time = self.peak_time
        peak_runoff = self.peak_runoff
        # print(f"Input shapes: {get_shape(runoff_dist)} x {get_shape([peak_time, peak_runoff])}", file=sys.stderr, flush=True)
        _vars = [runoff_dist, peak_time, peak_runoff]
        # print(f"Var types: {[type(v).__name__ for v in _vars]}", file=sys.stderr, flush=True)
        try:
            pair = [self.peak_time, self.peak_runoff]
            # print("Setup pair 1", file=sys.stderr, flush=True)
            hydrograph = self.runoff_dist * pair
        except:
            try:
                pair = [self.peak_time, self.peak_runoff]
                print("Initial pair", file=sys.stderr, flush=True)
                pair = np.array(pair)
                print("Setup pair 2", file=sys.stderr, flush=True)
                hydrograph = self.runoff_dist * pair
            except Exception as e:
                raise e
        # print(f"Hydrograph shape: {get_shape(hydrograph)}", file=sys.stderr, flush=True)
        hydrograph = distributions.increment(hydrograph, interval)
        return hydrograph

    def flood_data(self, rain_depths, interval):
        """Generate pairs of basin runoff flow generated from rainfall over time.

        Args:
            rain_depths (numpy.ndarray): A 2D array of scaled rainfall depths over time.
            interval (float): The amount of time the output will increment by.

        Yields:
            Tuple[float, float]: The next pair of time and runoff flow generated from rainfall.

        """
        rd = self.unit_hydrograph(interval).tolist()
        ri = list(self.runoff_depth_incremental(rain_depths, interval))
        ri.reverse()  # Reversed list utilized for synthesis
        comp_length = len(rd) + len(ri)
        for i in range(comp_length - 1):
            upper = i + 1
            lower = max(upper - len(ri), 0)
            total = sum(ri[j - upper] * rd[j][1] for j in range(lower, upper) if j < len(rd))
            yield i * interval, total

    def flood_hydrograph(self, rain_depths, interval):
        """Get a composite hydrograph of basin runoff generated from rainfall over time.

        Args:
            rain_depths (numpy.ndarray): A 2D array of scaled rainfall depths over time.
            interval (float): The amount of time the output will increment by.

        Returns:
            numpy.ndarray: The composite hydrograph of runoff generated from rainfall.

        """
        data = list(self.flood_data(rain_depths, interval))
        hydrograph = np.array(data)
        return hydrograph

class unit_hydrograph_model:
    default_curve:np.ndarray = np.array([
        [0.0, 0.0], [0.1, 0.03], [0.2, 0.1], [0.3, 0.19], [0.4, 0.31], [0.5, 0.47], [0.6, 0.66], 
        [0.7, 0.82], [0.8, 0.93], [0.9, 0.99], [1.0, 1.0], [1.1, 0.99], [1.2, 0.93], [1.3, 0.86], 
        [1.4, 0.78], [1.5, 0.68], [1.6, 0.56], [1.7, 0.46], [1.8, 0.39], [1.9, 0.33], [2.0, 0.28], 
        [2.2, 0.207], [2.4, 0.147], [2.6, 0.107], [2.8, 0.077], [3.0, 0.055], [3.2, 0.04], 
        [3.4, 0.029], [3.6, 0.021], [3.8, 0.015], [4.0, 0.011], [4.5, 0.005], [5.0, 0.0], 
    ])
    cn:float = 83.0
    tc:float = 2.3
    peak_factor:int = 1
    interval:float = 0.1
    basin:Basin
    flow_curve:np.ndarray
    empty_curve:np.ndarray
    def __init__(self, area:float = 0, duration:float = 24.0, interval:float = 0.1):
        if not isinstance(area, (int, float)):
            raise ValueError("Area must be a number.")
        if not isinstance(duration, (float)):
            raise ValueError("Duration must be a number.")
        if not isinstance(interval, (float)):
            raise ValueError("Interval must be a number.")
            
        self.basin = Basin(
            area=area,
            cn=self.cn,
            tc=self.tc,
            runoff_dist=self.default_curve,
            peak_factor=self.peak_factor
        )
        self.duration = duration
        self.interval = interval
        array_length = int(duration / interval) + 1
        array_timesteps = array_length
        array_length = max(array_length, self.default_curve.shape[0])
        # self.empty_curve = np.ndarray((int(duration / interval), 2))
        # self.empty_curve[:,0] = np.arange(0, duration, interval)
        # self.empty_curve[:,1] = np.zeros(int(duration / interval))
        self.empty_curve = np.zeros((array_length, 2))
        full_duration = np.ceil(array_timesteps * interval)
        self.empty_curve[:,0] = np.linspace(0, full_duration, array_length)
        self.flow_curve = self.empty_curve.copy()
        self.flow_curve[:,1] = np.zeros(array_length)
        self.timestep = 0
        print(f"Initialized unit hydrograph model with area {area} and duration {duration}.\nCurve shapes are default={self.default_curve.shape}, empty={self.empty_curve.shape}, flow={self.flow_curve.shape}", file=sys.stderr, flush=True)

    def get_input_curve(self, flow:float = 0)->np.ndarray:
        input_arr = self.empty_curve.copy()
        input_arr[self.timestep, 1] = flow
        return input_arr
    
    def step(self, flow:float = 0.0, time:float = 0.0)->float:
        # self.timestep = int(time) if time >= 0 and time <= self.duration else self.timestep
        # if flow > 0:
        #     print(f"[{self.timestep}|{time}]Flow input: {flow}", file=sys.stderr, flush=True)
        # print(f"{self.timestep} : {time}", file=sys.stderr, flush=True)
        if self.timestep >= self.flow_curve.shape[0]:
            return self.flow_curve[-1, 1]
        in_arr = self.get_input_curve(flow)
        # print(f"Inputs for flood_hydrograph: {in_arr}, {self.interval}", file=sys.stderr, flush=True)
        out_arr = self.basin.flood_hydrograph(in_arr, self.interval)
        out_arr = out_arr[:self.flow_curve.shape[0]]
        total_rainfall = in_arr[:,1].sum()
        total_runoff = out_arr[:,1].sum()
        # if total_runoff > 0 or total_rainfall > 0:
        #     print(f"Total rainfall: {total_rainfall}, Total runoff: {total_runoff}", file=sys.stderr, flush=True)
        self.flow_curve += out_arr
        self.timestep += 1
        if self.timestep >= self.flow_curve.shape[0]:
            return self.flow_curve[-1, 1]
        return self.flow_curve[self.timestep, 1]
            






if __name__ == "__main__":
    model = unit_hydrograph_model()
