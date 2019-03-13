import pandas as pd
import numpy as np

def trans2feature(data):
    future_name = ["period", "avg_voltage","terminate_voltage", "min_temp", "max_temp", "avg_temp", 
               "avg_voltage_load", "terminate_voltage_load", "begin_capacity", "capacity"]
    out = []
    period = 0
    keys = list(data.keys())
    begin_capacity = data[keys[0]]["Capacity"][0]
    for key in keys:
        one_period = data[key]
        avg_voltage = np.average(one_period["voltage_battery"])
        terminate_voltage = min(one_period["voltage_battery"])
        min_temp = min(one_period["temp_battery"])
        max_temp = max(one_period["temp_battery"])
        avg_temp = np.average(one_period["temp_battery"])
        avg_voltage_load = np.average(one_period["voltage_load"])
        i = len(one_period["voltage_load"]) - 1
        while one_period["voltage_load"][i] < 0.1:
            i = i - 1
        terminate_voltage_load = one_period["voltage_load"][i]
        capacity = one_period["Capacity"][0]
        one_data = [period, avg_voltage, terminate_voltage, min_temp, max_temp, avg_temp, avg_voltage_load, terminate_voltage_load, 
                    begin_capacity, capacity]
        out.append(one_data)
        period = period + 1
    return pd.DataFrame(out, columns = future_name)