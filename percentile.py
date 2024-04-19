#!/usr/bin/env python
# coding: utf-8

# In[17]:

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
HCPA_url = 'https://raw.githubusercontent.com/FardinSamadikh/content/main/HCPA_Metadata.csv'
HCPD_url = 'https://raw.githubusercontent.com/FardinSamadikh/content/main/HCPD.csv'
HCPY_url = 'https://raw.githubusercontent.com/FardinSamadikh/content/main/HCPY.csv'
parc_url = 'https://raw.githubusercontent.com/FardinSamadikh/content/main/pvs_clean.wmparc.csv'
PVS_url = 'https://raw.githubusercontent.com/FardinSamadikh/content/main/pvs_clean.wm.csv'



# Read HCPA 
meta_HCPA = fetch_data(HCPA_url)
meta_HCPA['interview_age'] = pd.to_numeric(meta_HCPA['interview_age'], errors='coerce')
meta_HCPA = meta_HCPA.drop(0)
#meta_HCPA.head(5)

# Read HCPD 
meta_HCPD = fetch_data(HCPD_url)
meta_HCPD=meta_HCPD.drop(0)
# Convert to numeric
meta_HCPD['interview_age'] = pd.to_numeric(meta_HCPD['interview_age'], errors='coerce')
meta_HCPD['interview_age_years']=meta_HCPD['interview_age']/12
#meta_HCPD.head(5)


# Read HCPY 
meta_HCPY = fetch_data(HCPY_url)
meta_HCPY['SubjectID'] = meta_HCPY['SubjectID'].str.replace('_', '')
meta_HCPY['Age_in_Yrs'] = pd.to_numeric(meta_HCPY['Age_in_Yrs'], errors='coerce')        
#meta_HCPY.head(5)


# Read PVS values in different regions 
parc = fetch_data(parc_url)
parc['patient_id']=parc['patient_id'].str.split('_').str[0]
#parc.head(5)


# Read Totoal PVS and WM values
TotPVS = fetch_data(PVS_url)
TotPVS['Subject']=TotPVS['Subject'].str.split('_').str[0]
#TotPVS.head(5)


# merge HCPA 
merged_df = pd.merge(parc, meta_HCPA[['src_subject_id', 'interview_age_years', 'sex']], left_on='patient_id', right_on='src_subject_id', how='left')
# merge HCPD
merged_df2 = pd.merge(merged_df, meta_HCPD[['src_subject_id', 'interview_age_years', 'sex']], left_on='patient_id', right_on='src_subject_id', how='left')
merged_df2.drop(['src_subject_id_y','src_subject_id_x'], axis=1, inplace=True)
# unite the values of columns 
merged_df2['interview_age_years_x'].fillna(merged_df2['interview_age_years_y'], inplace=True)
merged_df2['sex_x'].fillna(merged_df2['sex_y'], inplace=True)
# remove unnecessary columns
merged_df2.drop(['interview_age_years_y','sex_y'], axis=1, inplace=True)
#rename the column name 
merged_df2.rename(columns={'interview_age_years_x': 'age', 'sex_x': 'sex'}, inplace=True)
# merge HCPY
merged_df3 = pd.merge(merged_df2, meta_HCPY[['SubjectID', 'Age_in_Yrs', 'Gender']], left_on='patient_id', right_on='SubjectID', how='left')
merged_df3.drop('SubjectID', axis=1, inplace=True)
# unite the values of columns 
merged_df3['age'].fillna(merged_df3['Age_in_Yrs'], inplace=True)
merged_df3['sex'].fillna(merged_df3['Gender'], inplace=True)
# remove unnecessary columns
merged_df3.drop(['Age_in_Yrs','Gender'], axis=1, inplace=True)
# merge Total PVS and WM volumes
merged_df4 = pd.merge(merged_df3,TotPVS[['Subject','pvs_vol', 'wm_vol']], left_on='patient_id', right_on='Subject', how='left')
merged_df4.drop('Subject',axis=1, inplace=True)
#merged_df3.head(5)

#Normaliztion 
exclude_columns = ["age", "sex", "patient_id"]
# Selecting only columns other than the excluded ones
columns_to_normalize = [col for col in merged_df4.columns if col not in exclude_columns]
# Dividing each column's values by the values in the 'wm_vol' column
merged_df4[columns_to_normalize] = merged_df4[columns_to_normalize].div(merged_df4['wm_vol'], axis=0)
# drop WM and pvs_vol values - sort according the age - rounding age value 
#merged_df4['age'] = merged_df4['age'].round()
merged_df4.drop(['pvs_vol','wm_vol'],axis=1, inplace=True)
merged_df4['age'] = pd.to_numeric(merged_df4['age'], errors='coerce')
merged_df4.sort_values(by='age', ascending=True, inplace=True)
#merged_df4['age'].isnull().sum()
#merged_df4.loc[merged_df4['age'].isna()]
merged_df4.dropna(subset=['age'], inplace=True)
# import matplotlib.pyplot as plt
# plt.scatter(merged_df4['age'],merged_df4['wm-lh-bankssts'])


regions_dict={}
for column in merged_df4.columns:
    # Check if the column name follows the pattern of left and right regions
    if column.startswith('wm-lh-'):
        region_name = column[len('wm-lh-'):]  # Extract the region name
        print('Im checking '+region_name)
        # Check if the corresponding right region exists
        right_region_column = [col for col in merged_df4.columns if 'wm-rh-' + region_name in col]
        if right_region_column:  # If list is not empty, right region column exists
            # Create a new DataFrame containing the left and right regions with the same name
            region_df = merged_df4[['patient_id', 'sex', 'age', column, right_region_column[0]]]
            # Rename the columns
            region_df.columns = ['patient_id', 'sex', 'age', 'left_region', 'right_region']
            # Store the DataFrame in the dictionary
            regions_dict[region_name] = region_df
        else:
            print('wm-rh-' + region_name+' doesnt find')

import pandas as pd
#Create an ExcelWriter object to write to a single Excel file
with pd.ExcelWriter('regions_data2.xlsx') as writer:
    #Iterate over each key-value pair in regions_dict
   for region_name, region_df in regions_dict.items():
        #Write each DataFrame to a separate sheet in the Excel file
       region_df.to_excel(writer, sheet_name=region_name, index=False)


# Calculate percentiles for each combination of brain region, sex, and PVS value
percentiles = {}
for region, df_region in regions_dict.items():
    for sex in ['F', 'M']:  # Female, Male
        for side in ['right_region', 'left_region']:
            key = f'{region}_{sex}_{side}'
            percentiles[key] = {}
            for age in range(8, 80):  # Ages from 1 to 100
                # Round age to the nearest integer
                rounded_age = round(age)
                pvs_values = df_region[(df_region['sex'] == sex) & (df_region['age'].round() == rounded_age)][side]
                if not pvs_values.empty:
                    # Calculate percentiles
                    percentiles[key][age] = np.percentile(pvs_values, [5, 25, 50, 75, 90])
                else:
                    # If no exact match for age, interpolate based on adjacent ages
                    ages = sorted(df_region[df_region['sex'] == sex]['age'].round().unique())
                    if ages:  # Check if ages list is not empty
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

 
                    
# Visualization
# Import necessary libraries
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import urllib.parse
import plotly.graph_objs as go
import base64
import tkinter as tk
from tkinter import filedialog
import os

# Initialize Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div([
    html.H1("Percentile Curves for PVS Values"),
    dcc.Dropdown(
        id='region-dropdown',
        options=[{'label': region.capitalize(), 'value': region} for region in regions_dict.keys()],
        value='bankssts',
        clearable=False
    ),
    dcc.Dropdown(
        id='sex-dropdown',
        options=[
            {'label': 'Female', 'value': 'F'},
            {'label': 'Male', 'value': 'M'}
        ],
        value='F',
        clearable=False
    ),
    dcc.Dropdown(
        id='side-dropdown',
        options=[
            {'label': 'Right', 'value': 'right_region'},
            {'label': 'Left', 'value': 'left_region'}
        ],
        value='right_region',
        clearable=False
    ),
    dcc.Graph(id='percentile-curve'),
    html.Div([
        dcc.Input(id='age-from', type='number', placeholder='From', min=0, step=1),
        dcc.Input(id='age-to', type='number', placeholder='To', min=0, step=1),
        html.Button('Download CSV', id='download-button')
    ]),
    html.A(html.Button('Download Selected Ages'), id='download-link', download='filtered_regions_data.xlsx', href='javascript:void(0);', style={'display': 'none'})
])

# Define callback for downloading selected ages
@app.callback(
    [Output('download-link', 'href'),
     Output('download-link', 'children')],
    [Input('download-button', 'n_clicks')],
    [State('age-from', 'value'),
     State('age-to', 'value')]
)
def download_selected_ages(n_clicks, age_from, age_to):
    if n_clicks and age_from is not None and age_to is not None:
        # Create a directory selector pop-up
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        folder = filedialog.askdirectory()  # Open a file dialog for selecting a directory
        if folder:
            # Write the filtered data to a new Excel file
            with pd.ExcelWriter(os.path.join(folder, 'filtered_regions_data.xlsx')) as writer:
                # Write each filtered DataFrame to a separate sheet in the new Excel file
                for sheet_name, filtered_df in filtered_sheets.items():
                    filtered_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Encode the new Excel file to base64 for download
            with open(os.path.join(folder, 'filtered_regions_data.xlsx'), 'rb') as file:
                data = file.read()
                href_data = "data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64," + base64.b64encode(data).decode()

            # Return the download link
            return href_data, "Download Selected Ages"

    return 'javascript:void(0);', "Download Selected Ages"

# Callback to update the plot
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
        colors = ['rgba(0, 255, 0, 0.7)', 'rgba(255, 0, 255, 0.5)', 'rgba(255, 255, 0, 0.7)', 'rgba(0, 255, 255, 0.7)', 'rgba(0, 0, 255, 0.7)']
        for i, percentile_label in enumerate(percentiles_labels):
            pvs_values = percentiles_values[:, i]
            spline = go.Scatter(x=ages, y=pvs_values, mode='lines', name=f'{percentile_label} Percentile - {sex}', line=dict(shape='spline', smoothing=1.3, color=colors[i]))
            markers = go.Scatter(x=ages, y=pvs_values, mode='markers', marker=dict(color=colors[i], size=5), showlegend=False)
            traces.extend([spline, markers])
            if i > 0:
                fill_trace = go.Scatter(
                    x=ages + ages[::-1],
                    y=np.concatenate([percentiles_values[:, i], percentiles_values[::-1, i - 1]]),
                    fill='tozerox',
                    fillcolor=colors[i],
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    showlegend=False,
                    name=f'{percentiles_labels[i]} to {percentiles_labels[i - 1]} Percentile'
                )
                traces.append(fill_trace)
    layout = go.Layout(
        title=f'Percentile Curves for {region.capitalize()} Region - {sex} - {side.capitalize()} PVS Values',
        xaxis=dict(title='Age'),
        yaxis=dict(title='PVS Normalized value'),
        hovermode='closest',
        legend=dict(orientation='h', x=0, y=-0.2)
    )
    return {'data': traces, 'layout': layout}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)


