import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Specify the directory to save figures
FIGURE_DIR = r'C:\DATA\Job\Causalens\Figures'
os.makedirs(FIGURE_DIR, exist_ok=True)  # Create the directory if it doesn't exist

######################################################################################
######################################################################################
#                          Hourly plots
######################################################################################
######################################################################################
# Define the function to plot average rides by hour
def plot_average_rides_by_hour(df):
    df['hour'] = pd.to_datetime(df['hour'], format='%d/%m/%Y %H:%M')
    df['hour_of_day'] = df['hour'].dt.hour
    rides_per_hour = df.groupby('hour_of_day')['departure'].mean().reset_index()
    sorted_rides = rides_per_hour.sort_values(by='departure', ascending=False)
    top_hours = sorted_rides.head(4)
    least_hours = sorted_rides.tail(4)
    
    # Set colors: top hours in red, least hours in blue, others in gray
    colors = []
    for hour in rides_per_hour['hour_of_day']:
        if hour in top_hours['hour_of_day'].values:
            colors.append('red')
        elif hour in least_hours['hour_of_day'].values:
            colors.append('blue')
        else:
            colors.append('gray')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rides_per_hour, x='hour_of_day', y='departure', palette=colors)
    
    plt.title('Average Rides by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Rides')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'average_rides_by_hour.png'))
    plt.show()

def plot_total_rides_by_hour(df):
    df['hour'] = pd.to_datetime(df['hour'], format='%d/%m/%Y %H:%M')
    df['hour_of_day'] = df['hour'].dt.hour
    rides_per_hour = df.groupby('hour_of_day')['departure'].sum().reset_index()
    sorted_rides = rides_per_hour.sort_values(by='departure', ascending=False)
    top_hours = sorted_rides.head(4)
    least_hours = sorted_rides.tail(4)
    
    # Set colors: top hours in red, least hours in blue, others in gray
    colors = []
    for hour in rides_per_hour['hour_of_day']:
        if hour in top_hours['hour_of_day'].values:
            colors.append('red')
        elif hour in least_hours['hour_of_day'].values:
            colors.append('blue')
        else:
            colors.append('gray')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=rides_per_hour, x='hour_of_day', y='departure', palette=colors)
    plt.title('Total Rides by Hour of the Day')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Total Rides')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'total_rides_by_hour.png'))
    plt.show()

######################################################################################
######################################################################################
#                          daily plots
######################################################################################
######################################################################################
def plot_avg_daily_rides_by_top_stations(df_start, df_end, top_n=15):
    daily_departures = df_start.groupby(['start_station_name', 'day'])['departure'].sum().reset_index()
    avg_daily_departures = daily_departures.groupby('start_station_name')['departure'].mean().reset_index(name='avg_departures_per_day')
    
    daily_arrival = df_end.groupby(['end_station_name', 'day'])['arrival'].sum().reset_index()
    avg_daily_arrival = daily_arrival.groupby('end_station_name')['arrival'].mean().reset_index(name='avg_arrival_per_day')
    
    combined_stats = pd.merge(avg_daily_departures, avg_daily_arrival, 
                               left_on='start_station_name', 
                               right_on='end_station_name', 
                               how='outer')
    top_stations = combined_stats.nlargest(top_n, 'avg_departures_per_day')
    melted_stats = top_stations.melt(id_vars=['start_station_name'], 
                                       value_vars=['avg_departures_per_day', 'avg_arrival_per_day'],
                                       var_name='ride_type', 
                                       value_name='average_rides')
    plt.figure(figsize=(12, 6))
    palette = {'avg_departures_per_day': 'blue', 'avg_arrival_per_day': 'red'}  
    sns.barplot(data=melted_stats, x='start_station_name', y='average_rides', hue='ride_type', 
                order=top_stations['start_station_name'], palette=palette)

    plt.title('Average Daily Rides by Top Stations (Yearly Average)')
    plt.xlabel('Station Name')
    plt.ylabel('Average Rides Per Day')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'avg_daily_rides_by_top_stations_yearly.png'))
    plt.show()

    combined_stats['net_ride'] = combined_stats['avg_arrival_per_day'] - combined_stats['avg_departures_per_day']
    
    top_stations = combined_stats.nlargest(5, 'net_ride')  
    least_stations = combined_stats.nsmallest(5, 'net_ride')  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    
    palette = 'blue'  
    # Plot for Top Stations
    sns.barplot(data=top_stations, x='start_station_name', y='net_ride', color=palette, ax=ax1)
    ax1.set_title('Top Stations - Average daily Net Rides')
    ax1.set_xlabel('Station Name')
    ax1.set_ylabel('Average Net Rides Per Day')
    ax1.tick_params(axis='x', rotation=75)
    
    palette = 'red'  
    # Plot for Least Stations
    sns.barplot(data=least_stations, x='start_station_name', y='net_ride', color=palette, ax=ax2)
    ax2.set_title('Least Stations - Average Daily Net Rides')
    ax2.set_xlabel('Station Name')
    ax2.set_ylabel('Average Net Rides Per Day')
    ax2.tick_params(axis='x', rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'avg_daily_net_rides_top_least_stations_yearly.png'))
    plt.show()    

def plot_avg_duration_by_top_stations(df,top_n=15):
    daily_duration = df.groupby(['start_station_name', 'day'])['avg_duration'].sum().reset_index()
    avg_duration_per_station = daily_duration.groupby('start_station_name')['avg_duration'].mean().reset_index(name='avg_duration_per_day')
    top_stations = avg_duration_per_station.sort_values(by='avg_duration_per_day', ascending=False).head(top_n)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_stations, x='start_station_name', y='avg_duration_per_day', order=top_stations['start_station_name'])
    plt.title('Average Ride Duration by Top Stations (Daily Average)')
    plt.xlabel('Station Name')
    plt.ylabel('Average Ride Duration (minutes)')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'avg_duration_by_top_stations_daily.png'))
    plt.show()

def plot_rides_over_time_daily(df):
    plt.figure(figsize=(12, 6))
    df_grouped = df.groupby('day')['departure'].sum().reset_index()
    plt.plot(df_grouped['day'], df_grouped['departure'])
    plt.title('Total Rides Over Time (Daily)')
    plt.xlabel('Date')
    plt.ylabel('Total Rides')
    plt.xticks(rotation=75)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rides_over_time_daily.png'))
    plt.show()

def plot_top_stations_daily_demand(df, top_n=5):
    station_total_rides = df.groupby(['start_station_name', 'day'])['net_ride'].sum().reset_index()
    top_stations = station_total_rides.sort_values(by='net_ride', ascending=False).head(top_n)
    df_top_stations = df[df['start_station_name'].isin(top_stations['start_station_name'])]
    colors = sns.color_palette("tab10", top_n) 
    markers = ['o', 's', '^', 'D', 'P']  
    
    plt.figure(figsize=(12, 8))
    
    for i, station in enumerate(df_top_stations['start_station_name'].unique()):
        station_data = df_top_stations[df_top_stations['start_station_name'] == station]
        plt.plot(station_data['day'], station_data['net_ride'], 
                 marker=markers[i], linestyle='', 
                 label=f'{station}', color=colors[i], markersize=7)

    plt.title('Daily Net Ride Demand of Top Bike Stations Throughout the Year')
    plt.xlabel('Day of the Year')
    plt.ylabel('Net Rides')
    plt.legend(title='Station Name')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'Daily_Net_Ride_Demand_of_Top_Bike_Stations.png'))
    plt.show()

def plot_average_rides_by_day_of_week(df):
    df['day'] = pd.to_datetime(df['day'], format='%d/%m/%Y')
    df['day_of_week'] = df['day'].dt.day_name()
    rides_per_day = df.groupby('day_of_week')['departure'].mean().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rides_per_day['day_of_week'] = pd.Categorical(rides_per_day['day_of_week'], categories=day_order, ordered=True)
    rides_per_day.sort_values('day_of_week', inplace=True)
    plt.figure(figsize=(12, 6))
    max_rides = rides_per_day['departure'].max()
    min_rides = rides_per_day['departure'].min()
    colors = ['red' if value == max_rides else 'blue' if value == min_rides else 'gray' 
              for value in rides_per_day['departure']]
    sns.barplot(data=rides_per_day, x='day_of_week', y='departure', palette=colors)
    plt.title('Average Rides by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Rides')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'average_rides_by_day_of_week.png'))
    plt.show()

def plot_total_rides_by_day_of_week(df):
    df['day'] = pd.to_datetime(df['day'], format='%d/%m/%Y')
    df['day_of_week'] = df['day'].dt.day_name()
    rides_per_day = df.groupby('day_of_week')['departure'].sum().reset_index()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    rides_per_day['day_of_week'] = pd.Categorical(rides_per_day['day_of_week'], categories=day_order, ordered=True)
    rides_per_day.sort_values('day_of_week', inplace=True)
    plt.figure(figsize=(12, 6))
    max_rides = rides_per_day['departure'].max()
    min_rides = rides_per_day['departure'].min()
    colors = ['red' if value == max_rides else 'blue' if value == min_rides else 'gray' 
              for value in rides_per_day['departure']]
    sns.barplot(data=rides_per_day, x='day_of_week', y='departure', palette=colors)
    plt.title('Total Rides by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Total Rides')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'total_rides_by_day_of_week.png'))
    plt.show()
    
######################################################################################
######################################################################################
#                          Monthly plots
######################################################################################
######################################################################################
# Add similar sampling functionality for monthly stats
def plot_total_rides_by_top_station_monthly(df_start, df_end, top_n=10):
    monthly_departures = df_start.groupby(['start_station_name', 'month'])['departure'].sum().reset_index()
    avg_monthly_departures = monthly_departures.groupby('start_station_name')['departure'].mean().reset_index(name='avg_departures_per_month')
    
    monthly_arrival = df_end.groupby(['end_station_name', 'month'])['arrival'].sum().reset_index()
    avg_monthly_arrival = monthly_arrival.groupby('end_station_name')['arrival'].mean().reset_index(name='avg_arrival_per_month')
    
    combined_stats = pd.merge(avg_monthly_departures, avg_monthly_arrival, 
                               left_on='start_station_name', 
                               right_on='end_station_name', 
                               how='outer')
    top_stations = combined_stats.nlargest(top_n, 'avg_departures_per_month')
    melted_stats = top_stations.melt(id_vars=['start_station_name'], 
                                       value_vars=['avg_departures_per_month', 'avg_arrival_per_month'],
                                       var_name='ride_type', 
                                       value_name='average_rides')
    plt.figure(figsize=(12, 6))
    palette = {'avg_departures_per_month': 'blue', 'avg_arrival_per_month': 'red'} 
    sns.barplot(data=melted_stats, x='start_station_name', y='average_rides', hue='ride_type', 
                order=top_stations['start_station_name'], palette=palette)

    plt.title('Average monthly Rides by Top Stations (Yearly Average)')
    plt.xlabel('Station Name')
    plt.ylabel('Average Rides Per Month')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'avg_monthly_rides_by_top_stations_yearly.png'))
    plt.show()

    combined_stats['net_ride'] = combined_stats['avg_arrival_per_month'] - combined_stats['avg_departures_per_month']
    
    top_stations = combined_stats.nlargest(5, 'net_ride')  
    least_stations = combined_stats.nsmallest(5, 'net_ride')  

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    
    palette = 'blue'  
    # Plot for Top Stations
    sns.barplot(data=top_stations, x='start_station_name', y='net_ride', color=palette, ax=ax1)
    ax1.set_title('Top Stations - Average Monthly Net Rides')
    ax1.set_xlabel('Station Name')
    ax1.set_ylabel('Average Net Rides Per Month')
    ax1.tick_params(axis='x', rotation=75)
    
    palette = 'red'  
    # Plot for Least Stations
    sns.barplot(data=least_stations, x='start_station_name', y='net_ride', color=palette, ax=ax2)
    ax2.set_title('Least Stations - Average Monthly Net Rides')
    ax2.set_xlabel('Station Name')
    ax2.set_ylabel('Average Net Rides Per Month')
    ax2.tick_params(axis='x', rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'avg_monthly_net_rides_top_least_stations_yearly.png'))
    plt.show()    

def plot_avg_duration_by_top_station_monthly(df, top_n=10):
    monthly_duration = df.groupby(['start_station_name', 'month'])['avg_duration'].mean().reset_index()
    avg_monthly_duration = monthly_duration.groupby('start_station_name')['avg_duration'].mean().reset_index(name='avg_duration_per_month')
    
    top_stations = avg_monthly_duration.sort_values(by='avg_duration_per_month', ascending=False).head(top_n)
    df_top = df[df['start_station_name'].isin(top_stations['start_station_name'])]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_stations, x='start_station_name', y='avg_duration_per_month', order=top_stations['start_station_name'])
    plt.title('Average Duration by Top Stations (Monthly)')
    plt.xlabel('Station Name')
    plt.ylabel('Average Duration (minutes)')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'avg_duration_by_top_stations_monthly.png'))
    plt.show()

def plot_rides_by_bike_type_monthly_I(df):
    plt.figure(figsize=(12, 6))
    bike_types = ['Electric Bikes', 'Classic Bikes']
    total_rides = [df['total_electric_bikes'].sum(), df['total_classic_bikes'].sum()]
    sns.barplot(x=bike_types, y=total_rides)
    plt.title('Total Rides by Bike Type (Monthly)', fontsize=14, fontweight='bold')
    plt.xlabel('Bike Type', fontsize=14, fontweight='bold')
    plt.ylabel('Total Rides', fontsize=14, fontweight='bold')
    plt.xticks(rotation=75, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rides_by_bike_type_monthly_I.png'))
    plt.show()

def plot_rides_by_bike_type_monthly_II(df):
    plt.figure(figsize=(8, 8))
    bike_types = ['Electric Bikes', 'Classic Bikes']
    total_rides = [df['total_electric_bikes'].sum(), df['total_classic_bikes'].sum()]
    wedges, texts, autotexts = plt.pie(total_rides, labels=bike_types, autopct='%1.1f%%', startangle=140)
    plt.title('Total Rides by Bike Type (Monthly)', fontsize=14, fontweight='bold')
    for text in texts:
        text.set_fontsize(14)
        text.set_fontweight('bold')
        
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rides_by_bike_type_monthly_II.png'))
    plt.show()

def plot_rides_by_member_type_monthly_I(df):
    plt.figure(figsize=(12, 6))
    member_types = ['Members', 'Casual']
    total_rides = [df['total_members'].sum(), df['total_casual'].sum()]
    sns.barplot(x=member_types, y=total_rides)
    plt.title('Total Rides by Member Type (Monthly)', fontsize=14, fontweight='bold')
    plt.xlabel('Member Type', fontsize=14, fontweight='bold')
    plt.ylabel('Total Rides', fontsize=14, fontweight='bold')
    plt.xticks(rotation=75, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rides_by_member_type_monthly_I.png'))
    plt.show()

def plot_rides_by_member_type_monthly_II(df):
    plt.figure(figsize=(8, 8))
    member_types = ['Members', 'Casual']
    total_rides = [df['total_members'].sum(), df['total_casual'].sum()]

    wedges, texts, autotexts = plt.pie(
        total_rides, 
        labels=member_types, 
        autopct='%1.1f%%', 
        startangle=140,
        textprops={'fontsize': 14, 'fontweight': 'bold'},  
        pctdistance=0.80  
    )
    
    for autotext in autotexts:
        autotext.set_fontsize(14)  
        autotext.set_fontweight('bold')  
    plt.title('Total Rides by Member Type (Monthly)', fontsize=16, fontweight='bold')  
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'rides_by_member_type_monthly_II.png'))
    plt.show()    

def plot_top_stations_monthly_demand(df, top_n=5):
    station_total_rides = df.groupby(['start_station_name', 'month'])['net_ride'].sum().reset_index()
    top_stations = station_total_rides.sort_values(by='net_ride', ascending=False).head(top_n)
    df_top_stations = df[df['start_station_name'].isin(top_stations['start_station_name'])]
    colors = sns.color_palette("tab10", top_n) 
    markers = ['o', 's', '^', 'D', 'P']   
    plt.figure(figsize=(12, 8))
    
    for i, station in enumerate(df_top_stations['start_station_name'].unique()):
        station_data = df_top_stations[df_top_stations['start_station_name'] == station]
        plt.plot(station_data['month'], station_data['net_ride'], 
                 marker=markers[i], linestyle='', 
                 label=f'{station}', color=colors[i], markersize=7)

    plt.title('Monthly Net Ride Demand of Top Bike Stations Throughout the Year')
    plt.xlabel('Month of the Year')
    plt.ylabel('Net Rides')
    plt.legend(title='Station Name')
    plt.xticks(rotation=75)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'Monthly_Net_Ride_Demand_of_Top_Bike_Stations.png'))
    plt.show()
    
def plot_station_heatmap_monthly(df_start, df_end, output_file="bike_station_heatmap_monthly.html"):

    avg_departures = df_start.groupby('start_station_name')['departure'].mean().reset_index(name='avg_departures_per_month')
    avg_arrivals = df_end.groupby('end_station_name')['arrival'].mean().reset_index(name='avg_arrival_per_month')
    
    combined_stats = pd.merge(avg_departures, avg_arrivals, left_on='start_station_name', right_on='end_station_name', how='outer')
    combined_stats['net_ride'] = combined_stats['avg_arrival_per_month'] - combined_stats['avg_departures_per_month']
    
    station_coords = df_start[['start_station_name', 'mean_start_lat', 'mean_start_lng']].drop_duplicates()
    combined_stats = pd.merge(combined_stats, station_coords, on='start_station_name', how='left')
    
    heat_data = combined_stats[['mean_start_lat', 'mean_start_lng', 'net_ride']].dropna().values.tolist()
    center = [combined_stats['mean_start_lat'].mean(), combined_stats['mean_start_lng'].mean()]
    m = folium.Map(location=center, zoom_start=12)
    
    HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)
    
    output_path = os.path.join(FIGURE_DIR, output_file)
    m.save(output_path)
    print(f"Monthly heatmap saved at {output_path}.")


def plot_classified_heatmap(df_start, df_end, output_file="bike_station_heatmap_classified.html"):
    # Grouping and calculating mean departures and arrivals
    avg_departures = df_start.groupby('start_station_name')['departure'].mean().reset_index(name='avg_departures_per_month')
    avg_arrivals = df_end.groupby('end_station_name')['arrival'].mean().reset_index(name='avg_arrival_per_month')
    
    # Merging averages, ensure to handle NaN values
    combined_stats = pd.merge(avg_departures, avg_arrivals, left_on='start_station_name', right_on='end_station_name', how='outer')
    combined_stats['avg_departures_per_month'].fillna(0, inplace=True)  # Fill NaNs with 0 for departures
    combined_stats['avg_arrival_per_month'].fillna(0, inplace=True)      # Fill NaNs with 0 for arrivals
    
    # Calculate net ride
    combined_stats['net_ride'] = combined_stats['avg_arrival_per_month'] - combined_stats['avg_departures_per_month']
    
    # Get station coordinates
    station_coords = df_start[['start_station_name', 'mean_start_lat', 'mean_start_lng']].drop_duplicates()
    combined_stats = pd.merge(combined_stats, station_coords, on='start_station_name', how='left')
    
    # Fill NaNs in coordinates
    combined_stats['mean_start_lat'].fillna(0, inplace=True)
    combined_stats['mean_start_lng'].fillna(0, inplace=True)
    
    # Define categories based on net ride
    high, low = combined_stats['net_ride'].quantile([0.75, 0.25])
    combined_stats['category'] = pd.cut(combined_stats['net_ride'], bins=[float('-inf'), low, high, float('inf')], 
                                          labels=['Low', 'Medium', 'High'], include_lowest=True)
    
    # Create folium map
    center = [combined_stats['mean_start_lat'].mean(), combined_stats['mean_start_lng'].mean()]
    m = folium.Map(location=center, zoom_start=12)
    
    # Prepare heat data for all stations
    heat_data = combined_stats[['mean_start_lat', 'mean_start_lng', 'net_ride']].dropna().values.tolist()
    
    # Add heatmap
    HeatMap(heat_data, radius=15, max_zoom=13).add_to(m)
    
    # Sort and select top 5 and bottom 5 stations based on net_ride
    top_stations = combined_stats.nlargest(5, 'net_ride')
    bottom_stations = combined_stats.nsmallest(5, 'net_ride')
    selected_stations = pd.concat([top_stations, bottom_stations])
    
    # Add markers for selected stations
    for _, row in selected_stations.iterrows():
        if row['net_ride'] >= 0:  # Top stations
            color = 'blue'
        else:  # Bottom stations
            color = 'red'
        
        popup_text = f"Station: {row['start_station_name']}<br>Net Ride: {row['net_ride']:.2f}"
        folium.Marker(
            location=[row['mean_start_lat'], row['mean_start_lng']],
            popup=popup_text,
            icon=folium.Icon(color=color)
        ).add_to(m)
    
    # Save the heatmap
    output_path = os.path.join(FIGURE_DIR, output_file)
    m.save(output_path)
    print(f"Heatmap with classification saved at {output_file}.")