from typing import List, Dict, Tuple, Any
from pyflo import system, distributions
from pyflo.nrcs import hydrology
import numpy as np
import sys, os
# from matplotlib import pyplot as plt

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
register_factor("m", "mm", 1000)

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

class Basin(hydrology.Basin):
    area:float
    cn:float
    tc:float
    runoff_dist:np.ndarray
    peak_time:float
    peak_factor:float
    # def unit_hydrograph(self, interval):
    #     """Get a hydrograph that represents the time-flow relationship per unit (inch) of depth.

    #     Args:
    #         interval (float): The amount of time the output will increment by.

    #     Returns:
    #         numpy.ndarray: The hydrograph of potential basin runoff.

    #     """
    #     runoff_dist = self.runoff_dist
    #     peak_time = self.peak_time
    #     peak_runoff = self.peak_runoff
    #     # print(f"Input shapes: {get_shape(runoff_dist)} x {get_shape([peak_time, peak_runoff])}", file=sys.stderr, flush=True)
    #     _vars = [runoff_dist, peak_time, peak_runoff]
    #     # print(f"Var types: {[type(v).__name__ for v in _vars]}", file=sys.stderr, flush=True)
    #     try:
    #         pair = [self.peak_time, self.peak_runoff]
    #         # print("Setup pair 1", file=sys.stderr, flush=True)
    #         hydrograph = self.runoff_dist * pair
    #     except:
    #         try:
    #             pair = [self.peak_time, self.peak_runoff]
    #             print("Initial pair", file=sys.stderr, flush=True)
    #             pair = np.array(pair)
    #             print("Setup pair 2", file=sys.stderr, flush=True)
    #             hydrograph = self.runoff_dist * pair
    #         except Exception as e:
    #             raise e
    #     # print(f"Hydrograph shape: {get_shape(hydrograph)}", file=sys.stderr, flush=True)
    #     hydrograph = distributions.increment(hydrograph, interval)
    #     return hydrograph

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
            # if total != 0:
            #     print(f"total: {total}", file=sys.stderr, flush=True)
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
    area:float
    cn:float = 100.0 #83.0
    tc:float = 2.3
    peak_factor:int = 1
    interval:float
    basin:Basin
    flow_curve:np.ndarray
    empty_curve:np.ndarray
    def __init__(self, area:float = 0, duration:float = 24.0, interval:float = 1.0):
        if not isinstance(area, (int, float)):
            raise ValueError("Area must be a number.")
        if not isinstance(duration, (int, float)):
            raise ValueError("Duration must be a number.")
        if not isinstance(interval, (int, float)):
            raise ValueError("Interval must be a number.")
        # given in sqkm, convert to acres
        area = area * FACTOR_CONVERSIONS["sqkm"]["acres"]
        self.area = area
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
        self.empty_curve[:,0] = np.arange(0, full_duration, interval)
        self.flow_curve = self.empty_curve.copy()
        self.flow_curve[:,1] = np.zeros(array_length)
        self.so_far = []
        self.timestep = 0
        print(f"Initialized unit hydrograph model with area {area} and duration {duration}.\nCurve shapes are default={self.default_curve.shape}, empty={self.empty_curve.shape}, flow={self.flow_curve.shape}", file=sys.stderr, flush=True)

    def get_input_curve(self, flow:float = 0)->np.ndarray:
        input_arr = self.empty_curve.copy()
        input_arr[self.timestep, 1] = flow
        input_arr[self.timestep:, 1] = flow
        input_arr[:,1] = input_arr[:,1] * self.interval
        return input_arr
    
    def step(self, flow:float = 0.0, time:float = 0.0, arr:bool = False)->float:
        # if self.timestep != 50:
        #     flow = 0
        # else:
        #     flow = 1000
        # self.timestep = int(time) if time >= 0 and time <= self.duration else self.timestep
        # if flow > 0:
        #     print(f"[{self.timestep}|{time}]Flow input: {flow}", file=sys.stderr, flush=True)
        # print(f"{self.timestep} : {time}", file=sys.stderr, flush=True)
        if self.timestep >= self.flow_curve.shape[0]:
            return self.flow_curve[-1, 1]
        multiplier = 1
        in_arr = self.get_input_curve(flow)
        in_arr[:,1] = in_arr[:,1] * multiplier
        # print(f"Inputs for flood_hydrograph: {in_arr}, {self.interval}", file=sys.stderr, flush=True)
        out_arr = self.basin.flood_hydrograph(in_arr, self.interval)
        out_arr[:,1] = out_arr[:,1] / multiplier
        # if arr:
        #     return out_arr
        out_arr = out_arr[:self.flow_curve.shape[0]]
        self.flow_curve[:,1] += out_arr[:,1]
        # print(np.all(self.flow_curve[:,0] == out_arr[:,0]), file=sys.stderr, flush=True)
        # print(f"{self.flow_curve[:10,0]} vs {out_arr[:10,0]}", file=sys.stderr, flush=True)
        self.timestep += 1
        if self.timestep >= self.flow_curve.shape[0]:
            if arr:
                return self.flow_curve[-1, 1], out_arr
            return self.flow_curve[-1, 1]
        if arr:
            # if np.count_nonzero(out_arr[:,1]) > 0:
            #     #print out_arr with standard float formatting
            #     print(readable_ndarray(out_arr), file=sys.stderr, flush=True)
            return self.flow_curve[self.timestep, 1], out_arr
        return self.flow_curve[self.timestep, 1]
    
class unit_hydrograph_model_2(unit_hydrograph_model):
    """Simplified version of the unit hydrograph model class. Instead of creating and summing many arrays, maintain a single array and update it in place."""
    default_curve:np.ndarray = unit_hydrograph_model.default_curve.copy()
    # default_curve[:,0] = default_curve[:,0] * 2
    def __init__(self, area:float = 0, duration:float = 24.0, interval:float = 1.0):
        super().__init__(area, duration, interval)
        self.flow_curve = self.empty_curve.copy()
        self.flow_curve[:,1] = np.zeros(self.flow_curve.shape[0])
        self.so_far = []
        self.timestep = 0
        print(f"Initialized unit hydrograph model 2 with area {self.area} and duration {duration}.\nCurve shapes are default={self.default_curve.shape}, empty={self.empty_curve.shape}, flow={self.flow_curve.shape}", file=sys.stderr, flush=True)
        
    def convert_runoff_to_flow(self, runoff:float)->float:
        """
        runoff: float
            The runoff depth in inches.
        flow: float
            The flow rate in m^3/s.
        """
        area_sqkm = self.area * FACTOR_CONVERSIONS["acres"]["sqkm"]
        area_sqm = area_sqkm * 1e6
        runoff_m = runoff * 0.0254
        flow = runoff_m * area_sqm / self.interval / 3600
        return flow
    
    def convert_rainfall_to_raindepth(self, rainfall:np.ndarray)->np.ndarray:
        """
        rainfall: np.ndarray
            The rainfall depth in mm.
        raindepth: np.ndarray
            The rainfall depth in inches.
        """
        raindepth = rainfall * 0.0393701
        return raindepth
        
        
    def step(self, flow:float = 0.0, time:float = 0.0, arr:bool = False)->float:
        if self.timestep >= self.flow_curve.shape[0]:
            return self.flow_curve[-1, 1]
        self.so_far.append(flow)
        self.flow_curve[self.timestep:, 1] += flow
        in_arr = self.flow_curve.copy()
        in_arr[:,1] = self.convert_rainfall_to_raindepth(in_arr[:,1])
        
        
        out_arr = self.basin.flood_hydrograph(in_arr, self.interval)
        
        res = out_arr[self.timestep, 1]
        res = self.convert_runoff_to_flow(res)
        self.timestep += 1
        out_arr_2 = out_arr.copy()
        out_arr_2 = self.convert_runoff_to_flow(out_arr_2)
        return res if not arr else (res, out_arr_2)
            


def build_smoothened_default_curve():
    model = unit_hydrograph_model()
    import pandas as pd, numpy as np
    import matplotlib.pyplot as plt
    default_curve:np.ndarray = np.array([
        [0.0, 0.0], [0.1, 0.03], [0.2, 0.1], [0.3, 0.19], [0.4, 0.31], [0.5, 0.47], [0.6, 0.66], 
        [0.7, 0.82], [0.8, 0.93], [0.9, 0.99], [1.0, 1.0], [1.1, 0.99], [1.2, 0.93], [1.3, 0.86], 
        [1.4, 0.78], [1.5, 0.68], [1.6, 0.56], [1.7, 0.46], [1.8, 0.39], [1.9, 0.33], [2.0, 0.28], 
        [2.2, 0.207], [2.4, 0.147], [2.6, 0.107], [2.8, 0.077], [3.0, 0.055], [3.2, 0.04], 
        [3.4, 0.029], [3.6, 0.021], [3.8, 0.015], [4.0, 0.011], [4.5, 0.005], [5.0, 0.0], 
    ])
    def offset_curve(offset:float, magnitude:float)->np.ndarray:
        curve = default_curve.copy()
        curve[:,0] = curve[:,0] + offset
        curve[:,1] = curve[:,1] * magnitude
        return curve
    def smooth_curve(curve:np.ndarray)->np.ndarray:
        """Manipulate the curve to make the interval between points more uniform."""
        diffs = np.diff(curve[:,0])
        min_diff = diffs.min()
        datapoints = [(curve[0,0], curve[0,1])]
        for i in range(1, curve.shape[0]):
            prev = curve[i - 1]
            curr = curve[i]
            diff = curr[0] - prev[0]
            if diff == min_diff:
                datapoints.append((curr[0], curr[1]))
            else:
                steps = int(diff / min_diff)
                for j in range(1, steps + 1):
                    interp = prev + (curr - prev) * (j / steps)
                    datapoints.append(interp)
        return np.array(datapoints)
    fig, ax = plt.subplots()
    ax.scatter(default_curve[:,0], default_curve[:,1], label="Default")
    smoothened = smooth_curve(default_curve)
    ax.scatter(smoothened[:,0], smoothened[:,1], label="Smoothed")
    ax.legend()
    fig.savefig("unit_hydrograph.png")
    print(default_curve.shape, smoothened.shape)
    print(smoothened)
    smoothened_pd = pd.DataFrame(smoothened, columns=["Time", "Q"])
    print(smoothened_pd)
    smoothened_pd.to_csv("default_curve_smooth.csv", index=False)
    smoothened_area = np.trapz(smoothened[:,1], smoothened[:,0])
    print(f"Area under curve: {smoothened_area}")
    default_area = np.trapz(default_curve[:,1], default_curve[:,0])
    print(f"Default area under curve: {default_area}")
    unit_smoothened = smoothened.copy()
    unit_smoothened[:,1] = unit_smoothened[:,1] / smoothened_area
    unit_default = default_curve.copy()
    unit_default[:,1] = unit_default[:,1] / default_area
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.scatter(unit_default[:,0], unit_default[:,1], label="Unit Default")
    ax.scatter(default_curve[:,0], default_curve[:,1], label="Default")
    ax.legend()
    fig.savefig("unit_hydrograph_unit.png")
    def multiline_indent(lines:List[str], indent:int=4)->List[str]:
        return [f"{' ' * indent}{line}" for line in lines]
    def format_array_nicely(arr_name:str, arr:np.ndarray)->str:
        lines = []
        lines.append("import numpy as np")
        lines.append(f"{arr_name}: np.ndarray = np.array([")
        items = [f"[{x[0]}, {x[1]}]" for x in unit_smoothened]
        for i, item in enumerate(items):
            if i < len(items) - 1:
                items[i] = f"{item},"
        interior_lines = []
        column_width = 90
        for i, item in enumerate(items):
            if len(interior_lines) == 0:
                interior_lines.append(item)
            else:
                last_line = interior_lines[-1]
                if len(last_line) + len(item) + 1 < column_width:
                    interior_lines[-1] += f" {item}"
                else:
                    interior_lines.append(item)
        lines.extend(multiline_indent(interior_lines, 4))
        lines.append("])")
        return "\n".join(lines)
    unit_smoothened = unit_smoothened.round(6)
    with open("uh484.py", "w") as f:
        output_str = format_array_nicely("unit_smoothened", unit_smoothened)
        f.write(output_str)
    print(f"Default curve written to file.")
        
        
def make_test_raindata_regular(timesteps: int = 100, rainperiod: int = 100, amplitude: float = 1.0, num_events: int = 5, event_length_factor: float = 0.5) -> np.ndarray:
    """
    Generate a simple test dataset of rainfall data.
    
    Parameters:
        timesteps (int): The number of timesteps to generate.
        interval (float): The interval of time between each timestep in hours.
        
    Returns:
        np.ndarray: A 1D array of rainfall depths.
    """
    num_distinct_events = num_events
    period = rainperiod / num_distinct_events
    times = np.arange(0, timesteps + 1)
    sin_period = np.pi * 2 / period
    event_length_factor_shift = np.sin(np.pi / 2 * event_length_factor)
    event_length_factor_mult =  1 - event_length_factor_shift
    raindata = np.sin(times * sin_period) - event_length_factor_shift
    raindata = np.maximum(0, raindata)
    return raindata * amplitude / event_length_factor_mult


def test_functionality():
    import numpy as np, pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    thisdir = Path.cwd()
    dist_dir = thisdir / "dist"
    imgdir = dist_dir / "img"
    imgdir.mkdir(parents=True, exist_ok=True)
    # model = unit_hydrograph_model()
    amplitude = 1
    raindata = make_test_raindata_regular(
        timesteps=100, 
        rainperiod=100, 
        amplitude=amplitude, 
        num_events=1,
        event_length_factor=0
        )
    num_gt_zero = np.count_nonzero(raindata)
    print(f"Number of timesteps with rainfall: {num_gt_zero}/{len(raindata)}")
    print(f"Max rainfall: {np.max(raindata)}")
    # exit(0)
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(raindata)
    ymax = max(np.max(raindata), amplitude) * 1.1
    ymin = min(np.min(raindata), 0) * 1.1
    ax.set_ylim(ymin, ymax)
    ax.set_title("Rainfall data")
    fig.savefig(imgdir / "rainfall_data.png")
    
    _flow = np.zeros(len(raindata) + 20)
    _given_rain = np.zeros(len(raindata) + 20)
    areasqkm = 15
    model = unit_hydrograph_model_2(area=15, duration=len(_flow), interval=1)
    out_arrs = []
    for i, _ in enumerate(_flow):
        rain = 0 if i >= len(raindata) else raindata[i]
        # rain = rain * 1000
        flow, out_arr = model.step(flow=rain, time=i, arr=True)
        out_arr = out_arr[:len(raindata)]
        out_arrs.append(out_arr)
        sum_flow = np.sum(out_arr[:,1])
        # if sum_flow > 0:
        #     print(f"{rain} -> {sum_flow}")
        _flow[i] = flow
        _given_rain[i] = rain
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(_flow, label="Flow")
    # ax.plot(_given_rain, label="Rainfall")
    ax.plot(model.flow_curve[:,1], label="Model")
    ax.set_title("Flow data")
    ax.legend()
    fig.savefig(imgdir / "flow_data.png")
    plt.close()
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_title("Output graphs")
    for i, out_arr in enumerate(out_arrs):
        ax.plot(out_arr[:,1], label=f"Step {i}")
    ax.legend()
    fig.savefig(imgdir / "output_data.png")
    
    # Flow + Rainfall
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(_flow, label="Flow")
    ax.plot(raindata, label="Rainfall")
    ax.plot(_given_rain, label="Given Rainfall", linestyle="--")
    model_flow = model.flow_curve[:,1]
    model_flow = model_flow[:len(_flow)]
    ax.plot(model_flow, label="Model", linestyle="-.")
    ax.set_title("Flow and Rainfall data")
    ax.legend()
    fig.savefig(imgdir / "flow_rainfall_data.png")
    plt.close()
    
    # Rainfall -> Outputs
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.plot(raindata, label="Rainfall")
    nonzero_raindatas = [[i, raindata[i]] for i in range(len(raindata)) if raindata[i] > 0]
    important_inds = [i for i, _ in nonzero_raindatas]
    ind_count = len(important_inds)
    readable_count = min(10, ind_count)
    skipsize = ind_count // readable_count
    for i, _ in nonzero_raindatas:
        if i % skipsize != 0:
            continue
        associated_out_arr = out_arrs[i]
        plotted = ax.plot(associated_out_arr[:,1], label=f"Step {i}")
        color = plotted[0].get_color()
        # ax.bar(i, raindata[i], color=color, alpha=0.5)
    ax.set_title("Rainfall and Output data")
    ax.legend()
    fig.savefig(imgdir / "rainfall_output_data.png")
    plt.close()
    
    # rainfall is in mm, flow is in m^3/s
    # convert rainfall from mm to m^3
    raindata_m = raindata * 1e-3
    areasqm = areasqkm * 1e6
    raindata_m3 = raindata_m * areasqm
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_title("Rainfall and Flow data")
    times = np.arange(0, len(raindata))
    times_2 = np.arange(0, len(_flow))
    ax2 = ax.twinx()
    # Tie secondary y-axis to rainfall data, with 0 at top, and max at bottom
    ax2.bar(times, raindata_m3, label="Rainfall (m^3)")
    ax.plot(_flow, label="Flow")
    ax.xaxis.set_label_text("Time (hours)")
    ax.yaxis.set_label_text("Flow (m^3)")
    ax.plot(out_arrs[-1][:,1], label="Last output")
    ax1lims = ax.get_ylim()
    ax2.set_ylim(ax1lims[1], ax1lims[0])
    ax.legend()
    fig.savefig(imgdir / "volume_rain_flow_data.png")
    
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_title("Rainfall and Flow data (Non Bar)")
    ax.plot(times_2, _flow, label="Flow")
    ax.plot(times, raindata_m3, label="Rainfall (m^3)", linestyle="--")
    ax.legend()
    ax.xaxis.set_label_text("Time (hours)")
    ax.yaxis.set_label_text("Flow (m^3)")
    fig.savefig(imgdir / "volume_rain_flow_data_nonbar.png")
    
    def running_sum(arr:np.ndarray)->np.ndarray:
        return np.cumsum(arr)
    
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_title("Running sum of Flow vs Rainfall")
    ax.plot(running_sum(raindata_m3), label="Rainfall")
    ax.plot(running_sum(_flow), label="Flow")
    ax.legend()
    ax.xaxis.set_label_text("Time (hours)")
    ax.yaxis.set_label_text("Flow (m^3)")
    fig.savefig(imgdir / "running_sum_rain_flow_data.png")
    
    fig, ax = plt.subplots()
    ax:plt.Axes
    ax.set_title("Basin unit hydrograph")
    ax.plot(model.basin.unit_hydrograph(1))
    fig.savefig(imgdir / "basin_unit_hydrograph.png")
    
    
        
        

    
    
if __name__ == "__main__":
    # build_smoothened_default_curve()
    test_functionality()
    # print("Unit hydrograph model test complete.")
    print(np.sum(unit_hydrograph_model.default_curve[:,1]))
    default_curve_original:np.ndarray = np.array([
        [0.0, 0.0], [0.1, 0.03], [0.2, 0.1], [0.3, 0.19], [0.4, 0.31], [0.5, 0.47], [0.6, 0.66], 
        [0.7, 0.82], [0.8, 0.93], [0.9, 0.99], [1.0, 1.0], [1.1, 0.99], [1.2, 0.93], [1.3, 0.86], 
        [1.4, 0.78], [1.5, 0.68], [1.6, 0.56], [1.7, 0.46], [1.8, 0.39], [1.9, 0.33], [2.0, 0.28], 
        [2.2, 0.207], [2.4, 0.147], [2.6, 0.107], [2.8, 0.077], [3.0, 0.055], [3.2, 0.04], 
        [3.4, 0.029], [3.6, 0.021], [3.8, 0.015], [4.0, 0.011], [4.5, 0.005], [5.0, 0.0], 
    ])
    print(np.sum(default_curve_original[:,1]))
    

    