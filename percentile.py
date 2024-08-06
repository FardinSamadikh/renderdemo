
"""
INI-IMG lab- Dr.Choupan
@author: /F 
"""

import pandas as pd
import numpy as np
import requests
import io
import os


# Read data from GitHub repositories
def fetch_data(url):
    github_pat = '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'
    headers = {"Authorization": f"token {github_pat}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.read_csv(io.StringIO(response.text))
    else:
        print("Failed to retrieve data:", response.status_code)

# URLs for data
HCP_L_url= 'https://raw.githubusercontent.com/FardinSamadikh/Files/main/COMBAT_pvs-vf-regions_adj.lh_02272023.csv'
HCP_R_url= 'https://raw.githubusercontent.com/FardinSamadikh/Files/main/COMBAT_pvs-vf-regions_adj.rh_02272023.csv'

# Read Dataset

HCP_lh = fetch_data(HCP_L_url)
HCP_rh = fetch_data(HCP_R_url)


HCP_lh.drop(columns=['AGE', 'SEX_M', 'SCANNER', 'SITE'], inplace=True)
HCP_rh.drop(columns=['SCANNER', 'SITE'], inplace=True)

Merged = pd.merge(HCP_rh, HCP_lh, left_on='ID', right_on='ID', how='left')

Merged['AGE'] = pd.to_numeric(Merged['AGE'], errors='coerce')
Merged.sort_values(by='AGE', ascending=True, inplace=True)
Merged.dropna(subset=['AGE'], inplace=True)


Merged.rename(columns={'SEX_M': 'sex'}, inplace=True)
Merged.rename(columns={'AGE': 'age'}, inplace=True)

# Replace numerical values with 'F' and 'M' in the 'SEX' column
Merged['sex'] = Merged['sex'].replace({0: 'F', 1: 'M'})

# Function to rename columns
def rename_columns(col_name):
    if col_name.startswith('lh.'):
        return 'wm-lh-' + col_name[3:]
    elif col_name.startswith('rh.'):
        return 'wm-rh-' + col_name[3:]
    return col_name

# Apply the function to rename columns
Merged.columns = [rename_columns(col) for col in Merged.columns]

# Create an empty dictionary to store the regions
regions_dict = {}

# Iterate through the columns of the Merged dataframe
for column in Merged.columns:
    # Check if the column name follows the pattern of left and right regions
    if column.startswith('wm-lh-'):
        # Extract the region name
        region_name = column[len('wm-lh-'):]  
        print('Im checking '+region_name)
        
        # Check if the corresponding right region exists
        right_region_column = [col for col in Merged.columns if 'wm-rh-' + region_name in col]
        
        # If list is not empty, right region column exists
        if right_region_column:  
            # Create a new DataFrame containing the left and right regions with the same name
            region_df = Merged[['ID', 'sex', 'age', column, right_region_column[0]]]
            # Rename the columns
            region_df.columns = ['ID', 'sex', 'age', 'left_region', 'right_region']
            # Store the DataFrame in the dictionary
            regions_dict[region_name] = region_df
        else:
            print('wm-rh-' + region_name+' doesnt find')

#Create an ExcelWriter object to write to a single Excel file
with pd.ExcelWriter('regions_data2.xlsx') as writer:
    #Iterate over each key-value pair in regions_dict
   for region_name, region_df in regions_dict.items():
        #Write each DataFrame to a separate sheet in the Excel file
       region_df.to_excel(writer, sheet_name=region_name, index=False)


# Function to remove outliers using the IQR method
def remove_outliers(data):
    if len(data) < 2:
        return data  # Not enough data to compute IQR
    Q1 = np.percentile(data, 20)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

# Calculate percentiles for each combination of brain region, sex, and PVS value
percentiles = {}
for region, df_region in regions_dict.items():
    for sex in ['F', 'M']:  # Female, Male
        for side in ['right_region', 'left_region']:
            key = f'{region}_{sex}_{side}'
            percentiles[key] = {}
            for age in range(8, 85):  # Ages from 8 to 84
                rounded_age = round(age)
                pvs_values = df_region[(df_region['sex'] == sex) & (df_region['age'].round() == rounded_age)][side]
                
                if not pvs_values.empty:
                    # Remove outliers before calculating percentiles
                    pvs_values_clean = remove_outliers(pvs_values)
                    
                    if len(pvs_values_clean) > 0:
                        # Calculate percentiles
                        percentiles[key][age] = np.percentile(pvs_values_clean, [5,25, 50, 75, 90])
                    else:
                        # If all values are outliers, set to None
                        percentiles[key][age] = None
                else:
                    # If no exact match for age, interpolate based on adjacent ages
                    ages = sorted(df_region[df_region['sex'] == sex]['age'].round().unique())
                    if ages:
                        if rounded_age < min(ages):
                            min_age = min(ages)
                            if min_age in percentiles[key]:
                                percentiles[key][age] = percentiles[key][min_age]
                            else:
                                percentiles[key][age] = None
                        elif rounded_age > max(ages):
                            max_age = max(ages)
                            if max_age in percentiles[key]:
                                percentiles[key][age] = percentiles[key][max_age]
                            else:
                                percentiles[key][age] = None
                        else:
                            prev_age = max(a for a in ages if a < rounded_age)
                            next_age = min(a for a in ages if a > rounded_age)
                            prev_percentile = percentiles[key].get(prev_age, None)
                            next_percentile = percentiles[key].get(next_age, None)
                            if prev_percentile is not None and next_percentile is not None:
                                # Linear interpolation between adjacent percentiles
                                interpolated_percentile = np.interp(rounded_age, [prev_age, next_age], [prev_percentile, next_percentile])
                                percentiles[key][age] = interpolated_percentile
                            else:
                                # If adjacent percentiles are not available, use the nearest one
                                if prev_percentile is None:
                                    percentiles[key][age] = next_percentile
                                else:
                                    percentiles[key][age] = prev_percentile
                    else:
                        percentiles[key][age] = None
# HTML 
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.interpolate import UnivariateSpline
import subprocess
import logging
import os

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Percentile Curves for PVS Values"),
    html.Div([
        html.Div([
            html.Label("White Matter Region"),
            dcc.Dropdown(
                id='region-dropdown',
                options=[{'label': region.capitalize(), 'value': region} for region in regions_dict.keys()],
                value='bankssts',
                clearable=False
            )
        ], style={'display': 'inline-block', 'width': '32%', 'padding': '10px'}),
        html.Div([
            html.Label("Sex"),
            dcc.Dropdown(
                id='sex-dropdown',
                options=[
                    {'label': 'Female', 'value': 'F'},
                    {'label': 'Male', 'value': 'M'}
                ],
                value='F',
                clearable=False
            )
        ], style={'display': 'inline-block', 'width': '32%', 'padding': '10px'}),
        html.Div([
            html.Label("Hemisphere"),
            dcc.Dropdown(
                id='side-dropdown',
                options=[
                    {'label': 'Right', 'value': 'right_region'},
                    {'label': 'Left', 'value': 'left_region'}
                ],
                value='right_region',
                clearable=False
            )
        ], style={'display': 'inline-block', 'width': '32%', 'padding': '10px'}),
    ], style={'width': '100%', 'display': 'flex', 'justify-content': 'space-between'}),
    dcc.Graph(id='percentile-curve'),
    html.Div([
        html.Label("Download dataset: Enter the desired age range in the following fields", style={'font-weight': 'bold', 'font-size': '17px'}),
        html.Div([
            dcc.Input(id='age-from', type='number', placeholder='From', min=0, step=1),
            dcc.Input(id='age-to', type='number', placeholder='To', min=0, step=1),
            html.Button('Download CSV', id='download-button')
        ],style={'margin-top': '10px'}),
        html.A(html.Button('Download Selected Ages'), id='download-link', download='filtered_regions_data.xlsx', href='javascript:void(0);', style={'display': 'none'})
    ], style={'margin-top': '20px'})
])

def open_directory_selector(n_clicks):
    if n_clicks:
        try:
            output = subprocess.check_output(['python', 'directory_selector.py'])
            download_directory = output.decode().strip()
            return download_directory
        except Exception as e:
            print(f"Error: {e}")
            return ""
    else:
        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('download-link', 'href'),
     Output('download-link', 'children')],
    [Input('download-button', 'n_clicks')],
    [State('age-from', 'value'),
     State('age-to', 'value')]
)
def download_selected_ages(n_clicks, age_from, age_to):
    if n_clicks and age_from is not None and age_to is not None:
        try:
            filtered_data = Merged[(Merged['age'] >= age_from) & (Merged['age'] <= age_to)]

            download_directory = open_directory_selector(n_clicks)

            if download_directory:
                file_path = os.path.join(download_directory, 'filtered_regions_data.xlsx')
                filtered_data.to_excel(file_path, index=False)
                logging.debug(f"File saved to: {file_path}")
                return file_path, "Download Selected Ages"
            else:
                logging.debug("No folder selected.")
        except Exception as e:
            logging.error(f"Error creating file: {e}")

    return 'javascript:void(0);', "Download Selected Ages"

def smooth_data_spline(ages, data, s=1):
    spline = UnivariateSpline(ages, data, s=s)
    ages_smooth = np.linspace(ages.min(), ages.max(), 500)
    data_smooth = spline(ages_smooth)
    return ages_smooth, data_smooth

@app.callback(
    Output('percentile-curve', 'figure'),
    [Input('region-dropdown', 'value'),
     Input('sex-dropdown', 'value'),
     Input('side-dropdown', 'value')]
)
def update_plot(region, sex, side):
    traces = []
    key = f'{region}_{sex}_{side}'
    if key in percentiles:
        ages = np.array(list(percentiles[key].keys()))
        percentiles_values = np.array([val for val in percentiles[key].values() if val is not None])
        ages = np.array([age for age in ages if percentiles[key][age] is not None and 8 <= age <= 85])
        
        percentiles_values = percentiles_values[(ages >= 8) & (ages <= 85)]
        
        percentiles_labels = ['5th', '25th', '50th', '75th', '90th']
        colors = ['rgba(0, 255, 0, 0.7)', 'rgba(255, 0, 255, 0.5)', 'rgba(255, 255, 0, 0.7)', 'rgba(0, 255, 255, 0.7)', 'rgba(0, 0, 255, 0.7)']
        
        for i, percentile_label in enumerate(percentiles_labels):
            pvs_values = percentiles_values[:, i]
            ages_smooth, smoothed_values = smooth_data_spline(ages, pvs_values)
            
            spline = go.Scatter(x=ages_smooth, y=smoothed_values, mode='lines', name=f'{percentile_label} Percentile - {sex}', line=dict(shape='spline', smoothing=1.3, color=colors[i]))
            markers = go.Scatter(x=ages, y=pvs_values, mode='markers', marker=dict(color=colors[i], size=5), showlegend=False)
            traces.extend([spline, markers])
            
            if i > 0:
                fill_trace = go.Scatter(
                    x=np.concatenate([ages_smooth, ages_smooth[::-1]]),
                    y=np.concatenate([smoothed_values, smooth_data_spline(ages, percentiles_values[:, i - 1])[1][::-1]]),
                    fill='tozerox',
                    fillcolor=colors[i],
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    showlegend=False,
                    name=f'{percentiles_labels[i]} to {percentiles_labels[i - 1]} Percentile'
                )
                traces.append(fill_trace)

    layout = go.Layout(
        title=f'Percentile Curves for {region.capitalize()} Region - {sex} - {side.capitalize()} PVS Values',
        xaxis=dict(title='Age', range=[8, 85], tick0=8, dtick=5),  # Start ticks at 8 with a step of 5
        yaxis=dict(title='PVS Volume fraction value'),
        hovermode='closest',
        legend=dict(orientation='h', x=0, y=-0.2)
    )
    return {'data': traces, 'layout': layout}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)
