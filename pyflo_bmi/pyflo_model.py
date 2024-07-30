from typing import List, Dict, Tuple, Any
from pyflo import system
from pyflo.nrcs import hydrology
import numpy as np
# from matplotlib import pyplot as plt

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
    basin:hydrology.Basin
    flow_curve:np.ndarray
    empty_curve:np.ndarray
    def __init__(self, area:float = 0, duration:float = 24.0, interval:float = 0.1):
        self.basin = hydrology.Basin(
            area=area,
            cn=self.cn,
            tc=self.tc,
            runoff_dist=self.default_curve,
            peak_factor=self.peak_factor
        )
        self.duration = duration
        self.interval = interval
        self.empty_curve = np.ndarray((int(duration / interval), 2))
        self.empty_curve[:,0] = np.arange(0, duration, interval)
        self.empty_curve[:,1] = np.zeros(int(duration / interval))
        self.flow_curve = self.empty_curve.copy()
        self.timestep = 0

    def get_input_curve(self, flow:float = 0)->np.ndarray:
        input_arr = self.empty_curve.copy()
        input_arr[self.timestep, 1] = flow
        return input_arr
    
    def step(self, flow:float = 0)->float:
        in_arr = self.get_input_curve(flow)
        out_arr = self.basin.flood_hydrograph(in_arr, self.interval)
        self.flow_curve += out_arr
        self.timestep += 1
        return self.flow_curve[self.timestep, 1]






if __name__ == "__main__":
    model = unit_hydrograph_model()
