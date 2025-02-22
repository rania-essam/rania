 
# Data Visualization

# 1 - Line  Plot 
# Show temperature Change over time ( monthly maximum temperature changes )
import plotly.express as px

line_data = grouped_data['grouped_by_month']
line_data['Date'] = pd.to_datetime(line_data[['Year', 'Month']].assign(Day=1))
fig = px.line(line_data, x='Date', y='MaxTemp', title='Monthly Maximum Temperature Changes')
fig.show()   


# 2 - Area Plot
# compare cumulative rain across locations  ( sum of daily rain fall values for a location over a year)
import plotly.graph_objs as go

area_data = grouped_data['grouped_by_location']
fig = go.Figure()
fig.add_trace(go.Scatter(x=area_data['Location'], y=area_data['Rainfall'], mode='lines', fill='tozeroy'))
fig.update_layout(title='Cumulative Rainfall by Location', xaxis_title='Location', yaxis_title='Rainfall')
fig.show()


# 3 - Histogram
# visualize frequency of days with different rain fall levels.
rainfall = df['Rainfall']
fig = px.histogram(rainfall, x='Rainfall', nbins=30, title='Rainfall Distribution')
fig.update_layout(xaxis_title='Rainfall (mm)', yaxis_title='Frequency')
fig.show()



# 4 - Bar Chart
#Purpose: Compare average wind speeds across locations.
fig = px.bar(grouped_data['grouped_by_location'], x='Location', y='WindSpeed3pm', 
             title='Average Wind Speed at 3 PM by Location')
fig.update_layout(xaxis_title='Location', yaxis_title='Wind Speed (km/h)')
fig.show()


# 5 - Pie Chart
# rain vs non-rain days 
rain_counts = df['RainToday'].value_counts()
fig = px.pie(values=rain_counts, names=rain_counts.index, title='Rain vs. No Rain Days')
fig.show()

# 6 - Box Plot
# temperature changes across locations
fig = px.box(df, x='Location', y='MaxTemp', title='Temperature Variability by Location')
fig.update_layout(xaxis_title='Location', yaxis_title='Maximum Temperature')
fig.show()


# 7 - Scatter Plot
# relation between humidity and temperature at 3 PM.
fig = px.scatter(df, x='Humidity3pm', y='Temp3pm', color='RainToday', 
                 title='Humidity vs. Temperature at 3 PM')
fig.update_layout(xaxis_title='Humidity (%)', yaxis_title='Temperature (°C)')
fig.show()


# 8 - Bubble Plot
# Wind speed and direction visualized as bubbles 
fig = px.scatter(df, x='WindSpeed3pm', y='WindSpeed9am', size='WindGustSpeed', color='WindGustDir',
                 title='Wind Speeds with Gusts by Direction')
fig.update_layout(xaxis_title='Wind Speed at 9 AM', yaxis_title='Wind Speed at 3 PM')
fig.show()
