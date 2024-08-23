import os, sys, json, pickle
from pathlib import Path
from matplotlib import pyplot as plt
import math, random, numpy as np

import pyflo_bmi.bmi_pyflo as bmi_pyflo

output_var = "discharge_calculated"
# Load the model
model = bmi_pyflo.Bmi_Pyflo()

def init_model():
    catchment_area = 1.0
    model.initialize()
    model.set_value("area_sqkm", catchment_area)
    
def sim_data()->np.ndarray:
    n_steps = 720
    precip = np.zeros(n_steps)
    precip[0:720] = np.sqrt(np.arange(0, 720, 1) % 25)
    return precip

def send_receive_data(APCP_surface: float)->float:
    model.set_value("APCP_surface", APCP_surface)
    model.update()
    return model.get_value(output_var)

def run_model(precip:np.ndarray)->np.ndarray:
    init_model()
    n_steps = precip.size
    discharge = np.zeros(n_steps)
    for i in range(n_steps):
        discharge[i] = send_receive_data(precip[i])
        # print(f"Step {i}: Precipitation = {precip[i]}, Discharge = {discharge[i]}")
    return discharge

def plot_data(precip:np.ndarray, discharge:np.ndarray):
    plt.plot(precip, label="Precipitation", color="blue")
    plt.plot(discharge, label="Discharge", color="red")
    plt.legend()
    img_path = Path("dist/test_img/test_module.png")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path)
    
def main():
    precip = sim_data()
    discharge = run_model(precip)
    plot_data(precip, discharge)
    
if __name__ == "__main__":
    main()