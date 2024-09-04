from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional, TypeVar
import numpy as np
if __name__ == "__main__":
    from discrete_time_series import Discretime
    from conversions import FACTOR_CONVERSIONS
else:
    from .discrete_time_series import Discretime
    from .conversions import FACTOR_CONVERSIONS

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
            depth_mm (float): The depth, in millimeters.
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
            depth (float): The volume, in cubic meters.
        """
        if isinstance(depth, Discretime):
            if depth.data_unit != "mm":
                raise ValueError(f"Discretime data unit must be millimeters: {depth.data_unit}")
            depth.data_unit = "m3"
            
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
            interpolated (np.ndarray): The interpolated array.
        """
        if isinstance(array, Discretime):
            if array.interval == interval:
                return array.data.copy()
        return Discretime.discrete_interpolate(array, interval)
        
    def discrete_segment_area(self, curve:Discretime, index:int)->float:
        """
        Alternative method for segment_area_under_curve.
        Since the Discretime class enforces evenly-spaced data, 
        we can avoid working with x values until the end.

        Args:
            curve (Discretime): The curve to calculate the area under.
            index (int): The index of the segment to calculate the area under.
            
        Returns:
            area (float): The area under the segment of the curve.
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
    
    def discrete_ordinate_area(self, ordinate:Discretime)->float:
        """
        Alternative method for ordinate_area_under_curve.
        Since the Discretime class enforces evenly-spaced data, 
        we can avoid working with x values until the end.

        Args:
            ordinate (Discretime): The ordinate to calculate the area under the curve for.
            
        Returns:
            area (float): The area under the curve. 
                The ordinate is assumed to be unitless, so the area is in minutes..?
        """
        area = 0
        for i in range(ordinate.data.shape[0]-1):
            area += self.discrete_segment_area(ordinate, i)
        return area
    
    def discrete_unit_hydrograph(self, interval:int)->Discretime:
        """
        Generate a unit hydrograph for the basin.
        
        Args:
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            interpolated (Discretime): The generated unit hydrograph.
        """
        if interval == self.ordinates.interval:
            interpolated = self.ordinates.copy()
        else:
            ordinates = Discretime.discrete_interpolate(self.ordinates.data, interval)
            interpolated = Discretime(ordinates, time_unit="minutes", data_unit="unitless", interval=interval)
        area = self.discrete_ordinate_area(interpolated)
        interpolated /= area
        return interpolated
    
    def discrete_flood_data(self, rain_depths:Discretime, interval:int)->Discretime:
        """
        Generate pairs of basin runoff flow generated from rainfall over time.
        
        Args:
            rain_depths (Discretime): A 2D array of scaled rainfall depths over time in minutes and millimeters.
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            flood_data (Discretime): The generated pairs of time and runoff flow generated from rainfall,
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
            data_subset (Discretime): The generated pairs of time and runoff flow generated from rainfall,
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
        
    def discrete_flood_data_partials(self, rain_depths: Discretime, interval: int)->List[Discretime]:
        """
        Generate a list of flood curves created by individual rainfall events.
        
        Args:
            rain_depths (Discretime): A 2D array of scaled rainfall depths over time in minutes and millimeters.
            interval (int): The interval to increment calculations by, in minutes.
            
        Returns:
            partials (List[Discretime]): A list of generated pairs of time and runoff flow generated from rainfall,
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