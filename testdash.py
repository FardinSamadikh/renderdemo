#!/usr/bin/env python
# coding: utf-8

# In[16]:


import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import time

# Step 1: Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Step 2: Layout for collecting user information
user_info_layout = html.Div([
    html.Label('Name'),
    dcc.Input(id='name', type='text', placeholder='Enter your name'),
    
    html.Label('Email'),
    dcc.Input(id='email', type='email', placeholder='Enter your email'),
    
    html.Label('Company/Institution'),
    dcc.Input(id='company', type='text', placeholder='Enter your company/institution'),
    
    html.Button('Submit', id='submit-btn', n_clicks=0),
    html.Div(id='submit-message')
])

# Step 3: Layout for uploading CSV files and selecting features for plotting
upload_layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Br(),
    html.Label('Select features for plotting:'),
    dcc.Dropdown(
        id='feature-dropdown',
        multi=True
    ),
    dcc.RadioItems(
        id='plot-type-radio',
        options=[
            {'label': '2D Plot', 'value': '2d'},
            {'label': '3D Plot', 'value': '3d'}
        ],
        value='2d'
    ),
    html.Div(id='plot-output'),
    html.Label('Select the label feature for classification:'),
    dcc.Dropdown(
        id='label-feature-dropdown',
        multi=False
    ),
    html.Button('Compute Feature Importance', id='compute-importance-btn', n_clicks=0),
    dcc.Loading(id="loading-1", children=[html.Div(id="feature-importance-output")], type="default"),
    html.Button('Select Features and Train Model', id='select-train-btn', n_clicks=0),
    html.Div(id='train-message'),
    html.Div(id='model-training-results')
])

# Step 4: Callbacks
@app.callback(
    Output('submit-message', 'children'),
    Input('submit-btn', 'n_clicks'),
    State('name', 'value'),
    State('email', 'value'),
    State('company', 'value')
)
def submit_user_info(n_clicks, name, email, company):
    if n_clicks == 0:
        raise PreventUpdate
    # Your code to process user information goes here
    # For demonstration, simply return a success message
    return html.Div('User information submitted successfully!', style={'color': 'green'})

@app.callback(
    Output('output-data-upload', 'children'),
    Output('feature-dropdown', 'options'),
    Output('label-feature-dropdown', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    options = [{'label': col, 'value': col} for col in df.columns]
    return html.Div(f'File {filename} uploaded successfully!'), options, options

@app.callback(
    Output('plot-output', 'children'),
    Input('feature-dropdown', 'value'),
    Input('plot-type-radio', 'value'),
    State('upload-data', 'contents')
)
def plot_data(selected_features, plot_type, contents):
    if selected_features is None or plot_type is None or contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    if plot_type == '2d':
        if len(selected_features) != 2:
            return html.Div('Please select exactly 2 features for 2D plot')
        fig = px.scatter(df, x=selected_features[0], y=selected_features[1], title=f'2D Plot of {selected_features[0]} vs {selected_features[1]}')
    elif plot_type == '3d':
        if len(selected_features) != 3:
            return html.Div('Please select exactly 3 features for 3D plot')
        fig = px.scatter_3d(df, x=selected_features[0], y=selected_features[1], z=selected_features[2], title=f'3D Plot of {selected_features[0]}, {selected_features[1]} and {selected_features[2]}')
    return dcc.Graph(figure=fig)

@app.callback(
    Output('feature-importance-output', 'children'),
    Input('compute-importance-btn', 'n_clicks'),
    State('feature-dropdown', 'value'),
    State('label-feature-dropdown', 'value'),
    State('upload-data', 'contents')
)
def compute_feature_importance(n_clicks, selected_features, label_feature, contents):
    if n_clicks == 0 or selected_features is None or label_feature is None or contents is None:
        raise PreventUpdate
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    X = df[selected_features]
    y = df[label_feature]
    
    # Compute feature importance using LASSO
    clf = Lasso(alpha=0.1)
    clf.fit(X, y)
    importance_lasso = pd.DataFrame({'Feature': selected_features, 'Importance (LASSO)': clf.coef_})
    
    # Compute feature importance using Gini importance
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X, y)
    importance_gini = pd.DataFrame({'Feature': selected_features, 'Importance (Gini)': clf_rf.feature_importances_})
    
    importance_df = importance_lasso.merge(importance_gini, on='Feature')
    
    return html.Div([
        html.H5('Feature Importance Ratings'),
        html.Table([
            html.Thead(html.Tr([html.Th(col) for col in importance_df.columns])),
            html.Tbody([
                html.Tr([html.Td(importance_df.iloc[i][col]) for col in importance_df.columns])
                for i in range(len(importance_df))
            ])
        ])
    ])

@app.callback(
    [Output('train-message', 'children'),
     Output('model-training-results', 'children')],
    [Input('select-train-btn', 'n_clicks')],
    [State('feature-importance-output', 'children'),
     State('upload-data', 'contents'),
     State('feature-dropdown', 'value'),
     State('label-feature-dropdown', 'value')]
)
def select_features_and_train(n_clicks, feature_importance_output, contents, selected_features, label_feature):
    if n_clicks == 0 or feature_importance_output is None or contents is None or selected_features is None or label_feature is None:
        raise PreventUpdate
    
    # Extract df from contents
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Your code to allow user to select features and train model goes here
    start_time = time.time()
    
    # Simulate training process
    time.sleep(5)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Dummy results for demonstration
    X = df[selected_features]
    y = df[label_feature]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train)
    
    # Check if it's a binary classification problem
    if len(np.unique(y)) > 2:
        return ['Multiclass classification is not supported for ROC curve'], [None]
    
    # Compute ROC curve and AUC
    y_score = clf_rf.predict_proba(X_test)[:, 1]  # Use probability of the positive class for binary classification
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    roc_curve_plot = dcc.Graph(
        id='roc-curve',
        figure={
            'data': [
                {'x': fpr, 'y': tpr, 'type': 'scatter', 'mode': 'lines', 'name': 'ROC curve (area = %0.2f)' % roc_auc},
                {'x': [0, 1], 'y': [0, 1], 'type': 'line', 'color': 'navy', 'line_dash': 'dash', 'line_width': 1, 'opacity': 0.5}
            ],
            'layout': {
                'title': 'Receiver Operating Characteristic (ROC) Curve',
                'xaxis': {'title': 'False Positive Rate'},
                'yaxis': {'title': 'True Positive Rate'},
                'legend': {'x': 0, 'y': 1}
            }
        }
    )
    
    selected_features_list = html.Div([
        html.H5('Selected Features for Training:'),
        html.Ul([html.Li(feature) for feature in selected_features])
    ])
    
    num_classes = len(df[label_feature].unique())
    num_classes_display = html.Div(f'Number of Classes: {num_classes}')
    
    return ['Training successfully ended!', f'Estimated time: {elapsed_time:.2f} seconds', selected_features_list, num_classes_display, roc_curve_plot], [None, None]

# Step 5: Arrange layouts into Dash app
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='User Info', value='tab-1', children=[user_info_layout]),
        dcc.Tab(label='Upload Files', value='tab-2', children=[upload_layout]),
    ]),
])

# Step 6: Run the app
if __name__ == '__main__':
    app.run_server(port=8051)


# In[ ]:




