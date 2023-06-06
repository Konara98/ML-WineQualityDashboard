# Import relevant libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# Load dataset
data = pd.read_csv('data/winequality-red.csv')
# Check for missing values
data.isna().sum()
# Remove duplicate data
data.drop_duplicates(keep='first')
# Calculate the correlation matrix
corr_matrix = data.corr()
# Label quality into Good (1) and Bad (0)
data['quality'] = data['quality'].apply(lambda x: 1 if x >= 6.0 else 0)
# Drop the target variable
X = data.drop('quality', axis=1)
# Set the target variable as the label
y = data['quality']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
# Create an instance of the logistic regression model
logreg_model = LogisticRegression()
# Fit the model to the training data
logreg_model.fit(X_train, y_train)

# Create the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div(
    children=[
        html.H1('CO544-2023 Lab 3: Wine Quality Prediction', style={'textAlign': 'center'}),
        
        html.Div(
            children=[
                html.H3('Exploratory Data Analysis'),
                html.Label('Feature 1 (X-axis)'),
                dcc.Dropdown(
                    id='x_feature',
                    options=[{'label': col, 'value': col} for col in data.columns],
                    value=data.columns[0]
                )
            ],
            style={'width': '30%', 'display': 'inline-block', 'margin': '20px'}
        ),
        
        html.Div(
            children=[
                html.Label('Feature 2 (Y-axis)'),
                dcc.Dropdown(
                    id='y_feature',
                    options=[{'label': col, 'value': col} for col in data.columns],
                    value=data.columns[1]
                )
            ],
            style={'width': '30%', 'display': 'inline-block', 'margin': '20px'}
        ),
        
        dcc.Graph(id='correlation_plot'),
        
        # Wine quality prediction based on input feature values
        html.H3("Wine Quality Prediction", style={'marginTop': '50px'}),
        html.Div(
            children=[
                html.Label("Fixed Acidity"),
                dcc.Input(id='fixed_acidity', type='number', required=True),
                html.Label("Volatile Acidity"),
                dcc.Input(id='volatile_acidity', type='number', required=True),
                html.Label("Citric Acid"),
                dcc.Input(id='citric_acid', type='number', required=True),
                html.Br(),
                
                html.Label("Residual Sugar"),
                dcc.Input(id='residual_sugar', type='number', required=True),
                html.Label("Chlorides"),
                dcc.Input(id='chlorides', type='number', required=True),
                html.Label("Free Sulfur Dioxide"),
                dcc.Input(id='free_sulfur_dioxide', type='number', required=True),
                html.Br


