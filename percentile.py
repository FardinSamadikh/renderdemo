#!/usr/bin/env python
# coding: utf-8

# In[17]:

import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

# Generate random data
np.random.seed(42)  # Set seed for reproducibility

# Initialize empty dictionary to store data for each region
df_dict = {}

# Generate random data for each region
regions = ['posterior', 'inferior', 'amygdala', 'temporal', 'occipital']

for region in regions:
    # Generate random data for 1000 rows for each region
    age = np.random.randint(1, 101, size=1000)
    right_pvs_value = np.random.normal(loc=50, scale=10, size=1000)
    left_pvs_value = np.random.normal(loc=50, scale=10, size=1000)
    sex = np.random.choice(['f', 'm'], size=1000)
    
    # Create DataFrame for each region
    df_dict[region] = pd.DataFrame({'age': age, 'right_pvs_value': right_pvs_value, 'left_pvs_value': left_pvs_value, 'sex': sex})

# Calculate percentiles for each combination of brain region, sex, and PVS value
percentiles = {}
for region, df_region in df_dict.items():
    for sex in ['f', 'm']:  # Female, Male
        for side in ['right_pvs_value', 'left_pvs_value']:
            key = f'{region}_{sex}_{side}'
            percentiles[key] = {}
            for age in range(1, 101):  # Ages from 1 to 100
                pvs_values = df_region[(df_region['sex'] == sex) & (df_region[side].notna()) & (df_region['age'] == age)][side]
                if not pvs_values.empty:
                    percentiles[key][age] = np.percentile(pvs_values, [5, 25, 50, 75, 90])  # Percentiles

# Initialize Dash app
app = dash.Dash(__name__)
server=app.server
# Define layout
app.layout = html.Div([
    html.H1("Percentile Curves for PVS Values"),
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': region.capitalize(), 'value': region} for region in regions],
        value='posterior',
        clearable=False
    ),
    dcc.Dropdown(
        id='sex-dropdown',
        options=[
            {'label': 'Female', 'value': 'f'},
            {'label': 'Male', 'value': 'm'}
        ],
        value='f',
        clearable=False
    ),
    dcc.Dropdown(
        id='side-dropdown',
        options=[
            {'label': 'Right', 'value': 'right_pvs_value'},
            {'label': 'Left', 'value': 'left_pvs_value'}
        ],
        value='right_pvs_value',
        clearable=False
    ),
    dcc.Graph(id='percentile-curve')
])

# Define callback to update plot
@app.callback(
    Output('percentile-curve', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('side-dropdown', 'value')]
)
def update_plot(region, sex, side):
    traces = []
    key = f'{region}_{sex}_{side}'
    if key in percentiles:  # Ensure the key exists
        ages = list(percentiles[key].keys())
        percentiles_values = np.array(list(percentiles[key].values()))
        percentiles_labels = ['5th', '25th', '50th', '75th', '90th']
        for i, percentile_label in enumerate(percentiles_labels):
            pvs_values = percentiles_values[:, i]
            trace = go.Scatter(x=ages, y=pvs_values, mode='lines', name=f'{percentile_label} Percentile - {sex.upper()}')
            traces.append(trace)

    layout = go.Layout(
        title=f'Percentile Curves for {region.capitalize()} Region - {sex.upper()} - {side.capitalize()} PVS Values',
        xaxis=dict(title='Age'),
        yaxis=dict(title='PVS Value')
    )

    return {'data': traces, 'layout': layout}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)


