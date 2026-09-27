import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import folium

# Set the output directory
output_dir = r"C:\DATA\Job\Causalens\Analysis"
FIGURE_DIR = r'C:\DATA\Job\Causalens\Figures'
os.makedirs(output_dir, exist_ok=True)

import os

def analyze_departures_arrival_stats(df_start, df_end,time_unit, top_n, screen=False):

    departures = df_start.groupby(['start_station_name', time_unit])['departure'].sum().reset_index()
    avg_departures = departures.groupby('start_station_name')['departure'].mean().reset_index(name='avg_departures_per')

    # Find top and least N departure stations
    top_departure_stations = avg_departures.nlargest(top_n, 'avg_departures_per')
    least_departure_stations = avg_departures.nsmallest(top_n, 'avg_departures_per')

    arrival = df_end.groupby(['end_station_name', time_unit])['arrival'].sum().reset_index()
    avg_arrival = arrival.groupby('end_station_name')['arrival'].mean().reset_index(name='avg_arrival_per')

    # Find top and least N arrival stations
    top_arrival_stations = avg_arrival.nlargest(top_n, 'avg_arrival_per')
    least_arrival_stations = avg_arrival.nsmallest(top_n, 'avg_arrival_per')

    combined_stats = pd.merge(avg_departures, avg_arrival, 
                           left_on='start_station_name', 
                           right_on='end_station_name', 
                           how='outer')

    combined_stats['net_ride'] = combined_stats['avg_arrival_per'] - combined_stats['avg_departures_per']

    top_net_ride_stations = combined_stats.nlargest(top_n, 'net_ride')
    least_net_ride_stations = combined_stats.nsmallest(top_n, 'net_ride')

    top_departure_stations.to_csv(os.path.join(output_dir, f'{time_unit}_top_departure_stations.csv'), index=False)
    least_departure_stations.to_csv(os.path.join(output_dir, f'{time_unit}_least_departure_stations.csv'), index=False)
    top_arrival_stations.to_csv(os.path.join(output_dir, f'{time_unit}_top_arrival_stations.csv'), index=False)
    least_arrival_stations.to_csv(os.path.join(output_dir, f'{time_unit}_least_arrival_stations.csv'), index=False)
    top_net_ride_stations.to_csv(os.path.join(output_dir, f'{time_unit}_top_net_ride_stations.csv'), index=False)
    least_net_ride_stations.to_csv(os.path.join(output_dir, f'{time_unit}_least_net_ride_stations.csv'), index=False)

    if screen:
        print("Top Departure Stations:")
        print(top_departure_stations)
        print("\nLeast Departure Stations:")
        print(least_departure_stations)
        print("\nTop Arrival Stations:")
        print(top_arrival_stations)
        print("\nLeast Arrival Stations:")
        print(least_arrival_stations)
        print("\nTop Net Ride Stations:")
        print(top_net_ride_stations)
        print("\nLeast Net Ride Stations:")
        print(least_net_ride_stations)



