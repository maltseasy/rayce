from __future__ import annotations

from classes import *
from utils import load_csv
from constants import TILTS, SPEED_LIMITS
from sklearn.linear_model import LinearRegression
from pytz import timezone
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.stats import genhalflogistic
import datetime
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import folium
import branca.colormap
import pandas as pd

CELL_AREA = 0.0153 #m^2
air_density = 1.204 #kg/m^3
coef_drag = 0.19
coef_rr = 0.0023
car_mass = 575 #kg
accel_g = 9.81 #m/s^2
wheel_radius = 0.2 #m
cross_section = 2.21 #m^2
cen = timezone("US/Central")
est = timezone("US/Eastern")
utc = timezone("UTC")
    
def section_solar_power_out(coord: Checkpoint, time: datetime.datetime, section_tilts: dict):
    section_irradiance_sum = 0
    tilt_irradiances: dict[float, CellSolarData] = {}
    for array in section_tilts.keys():
        array_sum = 0
        for cell_angle in section_tilts[array]:
            if(cell_angle in tilt_irradiances):
                array_sum += tilt_irradiances[cell_angle].cell_power_out
            else:
                cell = CellSolarData(coord, time, cell_angle)
                tilt_irradiances[cell_angle] = cell
                array_sum += cell.cell_power_out
        section_irradiance_sum += array_sum
    return section_irradiance_sum #watts

def total_solar_power_out(coord: Checkpoint, time: datetime.datetime, tilts: dict):
    # iterating over all of the sections in the car
    car_power_sum: float = 0
    for section in tilts.keys():
        car_power_sum += section_solar_power_out(coord, time, tilts[section])
    #watts
    return car_power_sum

def energy_captured_between_points(c1: Checkpoint, current_time: datetime.datetime, segment_time: int):
    total_power_in_curr = total_solar_power_out(c1, current_time, tilts=TILTS)
    energy_this_segment = total_power_in_curr * segment_time / 3600 #wh
    return energy_this_segment

def energy_captured_along_route(time_initial: datetime.datetime, velocities: list[float], route: Route):
    current_time = time_initial
    total_energy = 0
    power_per_checkpoint = [0]

    for i in range(1, len(route.checkpoints)):
        old_speed = velocities[i-1] / 3.6
        new_speed = velocities[i] / 3.6
        v_avg_ms = (old_speed + new_speed) / 2

        segment_distance = route.checkpoints[i].distance - route.checkpoints[i-1].distance
        segment_time = segment_distance / v_avg_ms

        energy_this_segment = energy_captured_between_points(route.checkpoints[i-1], current_time, segment_time)
        current_time += datetime.timedelta(seconds=segment_time)
        power_per_checkpoint.append(energy_this_segment * 3600 / segment_time) 
        
        total_energy += energy_this_segment #watt hours
   
    # print("times: ", time_initial.astimezone(est), current_time.astimezone(est), (current_time - time_initial).seconds / 60)
    # print("avg power: ", total_energy / (current_time - time_initial).seconds)
    return total_energy, power_per_checkpoint #in watt hours, watts

def average_velocity(route: Route, velocities: list[float]):
    times = []
    total_time = 0
    for i in range(len(route.checkpoints)-1):
        if i > 0:
            avg_v = (velocities[i] + velocities[i-1])/2
            time = (route.checkpoints[i].distance - route.checkpoints[i-1].distance) / avg_v
            times.append(time)
            total_time += time
    v_avg = 0
    for i in range(len(route.checkpoints)-1):
        if i > 0:
            v_avg += velocities[i-1] * times[i-1] / total_time
    return v_avg


# pll = parallel
def calculate_work(grad: float, speed_initial: float, speed_end: float, pll_wind_speed: float, time: float):
    _speed_0 = speed_initial #m/s
    _speed_1 = speed_end #m/s
    accel = (_speed_1 - _speed_0) / time
    force_rr = coef_rr*car_mass*accel_g*math.cos(grad)
    force_drag = 0.5*air_density*coef_drag*((_speed_0 + _speed_1)/2 + pll_wind_speed)**2
    force_grad = car_mass*accel_g*math.sin(grad)
    force_accel = car_mass*accel
    accel_distance = time*_speed_0 + (accel*time**2)/2
    force_tractive = force_rr + force_drag + force_grad + force_accel
    work = accel_distance * force_tractive 

    return work #J


def predict_average_electrical_power(avg_tractive_power: float):
    efficiencyDataX = np.array([0, 16.688, 152.532, 246.05, 335.306, 423.94, 513.536, 677.019,
    840.528, 996.03, 1147.608, 1278.624, 1405.316, 1537.61, 1783.478, 1974.78])
    efficiencyDataY = np.array([0, 28, 171, 266, 359, 451, 544, 721, 898, 1071, 1242, 1402, 1558, 1718, 2069, 2385])
    
    degree = 3
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(efficiencyDataX.reshape(-1,1), efficiencyDataY)

    avg_electrical_power = polyreg.predict(np.array([avg_tractive_power]).reshape((-1,1)))
    return avg_electrical_power[0] #Watts

def solve_for_gradient(delta_elev, delta_d):
    c = math.sqrt(delta_d**2 + delta_elev**2)
    return math.asin(delta_elev/c)

# assumption: c1 and c2 are close together
def energy_used_between_points(c1: Checkpoint, c2: Checkpoint, speed_initial: float, speed_end: float, time: float):
    theta = solve_for_gradient(c2.elevation - c1.elevation, c2.distance - c1.distance)
    
    avg_azimuth = (c1.azimuth + c2.azimuth)/2
    dir_vec = Vec(avg_azimuth, 1)

    avg_wind_speed = (c1.wind_speed + c2.wind_speed)/2
    w1 = Vec(c1.wind_dir, c1.wind_speed)
    w2 = Vec(c2.wind_dir, c2.wind_speed)
    avg_w = w1.add(w2).normalize().mult(avg_wind_speed)

    pll_w = avg_w.proj(dir_vec)
    work = calculate_work(theta, speed_initial, speed_end, pll_w.magnitude, time)
    avg_tractive_power = work / time

    if avg_tractive_power <= 0:
        regen_efficiency = 0.05
        avg_electrical_power = avg_tractive_power * regen_efficiency
    else:
        avg_electrical_power = predict_average_electrical_power(avg_tractive_power)
    
    electrical_energy = avg_electrical_power * time
    passive_consumption = 10 #W
    electrical_energy += passive_consumption * time
    # print(f'theta: {math.degrees(theta)}, tractive: {avg_tractive_power}, elec: {avg_electrical_power}')
    return electrical_energy, avg_electrical_power, avg_tractive_power

def energy_used_along_route(route: Route, velocities: list[float]):
    total_energy = 0
    power_per_checkpoint = [0]
    for i in range(1, len(route.checkpoints)):
        old_speed = velocities[i-1] / 3.6
        new_speed = velocities[i] / 3.6
        v_avg_ms = (old_speed + new_speed) / 2
        time = (route.checkpoints[i].distance - route.checkpoints[i-1].distance) / v_avg_ms
        energy_used, avg_electrical_power, _ = energy_used_between_points(route.checkpoints[i-1], route.checkpoints[i], old_speed, new_speed, time)
        power_per_checkpoint.append(avg_electrical_power)
        total_energy += energy_used
    total_energy /= 3600
    return total_energy, power_per_checkpoint #wH

def solve_for_time(current_velocity: float, accel: float, distance: float):
    assert current_velocity**2 + 2*accel*distance < 0
    time = (-current_velocity + math.sqrt(current_velocity**2 + 2*accel*distance) )/ accel #s
    if time < 0:
        time = (-current_velocity - math.sqrt(current_velocity**2 + 2*accel*distance) )/ accel 
    return time

def average_velocity(route: Route, velocities: list[float]):
    times = []
    total_time = 0
    for i in range(len(route.checkpoints)-1):
        if i > 0:
            avg_v = (velocities[i] + velocities[i-1])/2
            time = (route.checkpoints[i].distance - route.checkpoints[i-1].distance) / avg_v
            times.append(time)
            total_time += time
    v_avg = 0
    for i in range(len(route.checkpoints)-1):
        if i > 0:
            v_avg += velocities[i-1] * times[i-1] / total_time
    return v_avg


fig, ax = plt.subplots(1,1)
a = 0.77
mean, var, skew, kurt = genhalflogistic.stats(a, moments='mvsk')
loc = -0.2
scale = 1.1
x = np.linspace(-1, 2, 120)
ax.plot(x, genhalflogistic.pdf(x,a,loc=loc, scale=scale), 'r-', lw=5, alpha=0.6, label='genhalflogistic pdf')
rv = genhalflogistic(a, loc=loc, scale=scale)
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = genhalflogistic.ppf([0.001, 0.5, 0.999], a, loc=loc,scale=scale)
np.allclose([0.001, 0.5, 0.999], genhalflogistic.cdf(vals, a, loc=loc, scale=scale))
r = genhalflogistic.rvs(a, size=1000, loc=loc, scale=scale)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.set_xlim([-1, 2])
ax.legend(loc='best', frameon=False)
plt.show()
#goal is to find acceleration that results in lowest power value.

def simulate_power_used(c1, c2, speed_initial, distance, current_speed_limit, trials=5):
    a = 0 #we know distance, need to find time
    score = 0
    v_f = 0
    current_trial = 0
    total_trials = 0
    while(current_trial < trials):
        r1 = genhalflogistic.rvs(0.77, size=1, loc=-0.35, scale=1.25)
        speed_range = current_speed_limit - speed_initial
        speed_final_sample = r1[0] * speed_range + speed_initial
        if(speed_final_sample < 0):
            continue
        a_calc = (speed_final_sample**2 - speed_initial**2) / (2*distance)
        try:
            time = 2*distance / (speed_final_sample + speed_initial)
            _, avg_elec_power, avg_tractive_power = energy_used_between_points(c1, c2, speed_initial, speed_final_sample, time)
            eff = avg_tractive_power / avg_elec_power if avg_elec_power > 0 else avg_elec_power / avg_tractive_power
            score_i = eff

            if score_i >= score:
                a = a_calc
                score = score_i
                v_f = speed_final_sample
                if score_i >= 0.3:
                    current_trial += 1
                total_trials +=1
                if(score_i > 0.85):
                    break
            else:
                total_trials +=1
            if total_trials > 50:
                break
            # print(f"Trial {current_trial}: Eff {round(eff,2) * 100}%, a_calc: {a_calc}, distance: {distance}, elec: {avg_elec_power}, trac: {avg_tractive_power}")
        except AssertionError:
            print(f"Trial {current_trial}: Invalid Samples, a_calc: {a_calc}, distance: {distance}")
            continue
        
    if score < 0.2:
        print(f"Total Trials {total_trials}. Score: {score}, a: {a}")
    return a, v_f

def simulate_energy_captured(c: Checkpoint, distance: float, current_time: datetime.datetime, speed_initial: float, current_speed_limit: float, trials=10):
    a = 0 #we know distance, need to find time
    score = 0
    v_f = 0
    current_trial = 0
    total_trials = 0
    t = 0
    while(current_trial < trials):
        r1 = genhalflogistic.rvs(0.77, size=1, loc=-0.35, scale=1.25)
        speed_range = current_speed_limit - speed_initial
        speed_final_sample = r1[0] * speed_range + speed_initial
        if(speed_final_sample <= 0):
            continue
        a_calc = (speed_final_sample**2 - speed_initial**2) / (2*distance)
        try:
            time = 2*distance / (speed_final_sample + speed_initial)
            energy_in = energy_captured_between_points(c, current_time, time)

            score_i: float = energy_in
            if score_i >= score:
                a = a_calc
                score = score_i
                v_f = speed_final_sample
                t = time
                current_trial += 1
                total_trials += 1
            else:
                total_trials += 1
            if total_trials > 50:
                break

            # print(f"Trial {current_trial}: Score {round(score_i,2)}, a_calc: {a_calc}, distance: {distance}")
        except AssertionError:
            print(f"Trial {current_trial}: Invalid Samples, a_calc: {a_calc}, distance: {distance}")
            continue
        
    print(f"Total Trials {total_trials}. Score: {score}, a: {a}")
    return a, v_f,t

# def simulate_velocities(c1: Checkpoint, c2: , distance: float, current_time: datetime.datetime, speed_initial:float, trials=10):
#     energy_factor = 0.1
#     speed_factor = 0.9
#     #distribution center = most eff v + route_dist_elapsed/2(v_target*2 - v_eff - v_avg) + solar_bias_factor
#     #find predicted min,max powers -> create solar forecast data -> norm = 7/20*cos(pi*s)+13/20

def assign_velocities(route: Route, time_initial:datetime.datetime, speed_limits: dict):
    current_time = time_initial
    #assert average_velocity > 35 mph
    velocities = [0]
    j = 0
    speed_limit_keys = list(speed_limits.keys())
    for i in range(1, len(route.checkpoints) - 1):
        if (j < len(speed_limits.keys())):
            speed_limit_ends_at = speed_limit_keys[j+1] * 1.60934
            if(route.checkpoints[i].distance / 1000 >= speed_limit_ends_at):
                j+=1
        if i == 0:
            velocities.append(0)
        else:
            current_speed_limit = speed_limits[speed_limit_keys[j]] * 1.60934 / 3.6#convert to m/s
            distance = route.checkpoints[i].distance - route.checkpoints[i-1].distance #meters
            current_velocity = velocities[i-1] / 3.6
            start = time.time()
            _, final_velocity = simulate_power_used(route.checkpoints[i-1], route.checkpoints[i], current_velocity, distance, current_speed_limit)
            _, final_velocity_2, _ = simulate_energy_captured(route.checkpoints[i-1], distance, current_time, current_velocity, current_speed_limit)
            end= time.time()
            weighted_v = final_velocity * 0.9 + final_velocity_2 * 0.1
            t = distance / weighted_v
            print(f'{(end-start)*1000} ms')
            current_time += datetime.timedelta(seconds=t)
            velocities.append(weighted_v * 3.6)
    velocities.append(0)
    return velocities

route = load_csv("A. Independence to Topeka")
loc_dt = datetime.datetime(2023, 6, 1, 8, 0, 0, tzinfo=est)
utc_dt = loc_dt.astimezone(utc)

velocities = assign_velocities(route, loc_dt, SPEED_LIMITS)
avg_v = average_velocity(route, velocities)
print("avg v: ", avg_v)

e1, power_outs = energy_used_along_route(route, velocities)
print("energy used (wh): ", e1)
e2, power_ins = energy_captured_along_route(utc_dt, velocities, route)
print("energy captured (wh): ", e2)
print(max(power_outs), min(power_outs), max(power_ins), min(power_ins))
print(len(route.checkpoints), len(power_ins), len(power_outs), len(velocities))


def route_to_list(route: Route):
    coords: list[tuple[float, float]] = [] 
    for checkpoint in route.checkpoints:
        coords.append((checkpoint.lat, checkpoint.lon))
    return coords

def create_map(route: Route, values: list, diverging = True):
    coords = route_to_list(route)
    m = folium.Map()
    if diverging:
        colors: list = branca.colormap.linear.RdYlGn_04.colors
        colors.reverse()
        colormap = branca.colormap.LinearColormap(colors=colors, vmin=min(values), vmax=max(values))
    else:
        colormap = branca.colormap.linear.YlGnBu_07.scale(min(values), max(values)).to_step(7)
    tooltip = [f"{str(v)} | Value: {round(values[i])}" for (i,v) in enumerate(route.checkpoints)]
    for i in range(len(coords)-1):
        rgba = colormap.rgba_floats_tuple(values[i])
        rgba_scaled = (round(i*255) for i in rgba[0:3])
        # print('#{:02x}{:02x}{:02x}'.format(*rgba_scaled))
        folium.PolyLine(locations=[coords[i], coords[i+1]], tooltip=tooltip[i], weight=10, color='#{:02x}{:02x}{:02x}'.format(*rgba_scaled)).add_to(m)
    # folium.ColorLine(positions=coords, colormap=colormap, weight=5, colors=power_list).add_to(m)
    # for i,p in zip_longest(power_list, coords, fillvalue=np.mean(power_list)):
    #     folium.Marker(p, tooltip = i).add_to(m)
    df = pd.DataFrame(coords).rename(columns={0: 'lat', 1: 'lon'})
    sw = df[['lat', 'lon']].min().values.tolist()
    ne = df[['lat', 'lon']].max().values.tolist()
    m.fit_bounds([sw, ne])
    m.add_child(colormap)
    return m
# m1 = create_map(route, powers_in)
m = create_map(route, velocities, False)
m