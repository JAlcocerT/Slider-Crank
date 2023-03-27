import dash
from dash import dcc
from dash import html

from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
import sympy as sp

from scipy.optimize import fsolve
from math import sin, cos, radians


def symbolic_rotation_matrix(angle_degrees_symbol):
    """
    Calculate the symbolic rotation matrix between two 2D coordinate systems given the symbolic angle between them.

    Args:
        angle_degrees_symbol (sympy.Symbol): The symbolic angle between the two coordinate systems in degrees.

    Returns:
        sympy.Matrix: The 2x2 symbolic rotation matrix.
    """
    angle_radians_symbol = sp.rad(angle_degrees_symbol)

    # Calculate the elements of the rotation matrix
    c = sp.cos(angle_radians_symbol)
    s = sp.sin(angle_radians_symbol)

    # Create the symbolic rotation matrix
    matrix = sp.Matrix([[c, -s],
                        [s, c]])

    return matrix


def slider_crank_position(l2, l3, offset, angle_deg_2):


    #Location of the joints
    u1_c_pr12x = 0
    u1_c_pr12y = 0
    u2_c_pr12x = 0
    u2_c_pr12y = 0

    u2_c_pr23x = l2
    u2_c_pr23y = 0
    u3_c_pr23x = 0
    u3_c_pr23y = 0

    u3_c_pr34x = l3
    u3_c_pr34y = 0
    u4_c_pr34x = 0
    u4_c_pr34y = 0

    alfa = 0
    h1x=0
    h1y=0
    u1_px = 0
    u1_py = 0
    u4_px = 0
    u4_py = 0

    # Define q - the symbolic angles and positions
    #angle_deg_1 = 0
    angle_deg_1 = sp.Symbol('theta_1')
    angle_deg_3 = sp.Symbol('theta_3')
    angle_deg_4 = sp.Symbol('theta_4')

    #Rx_1 = 0
    Rx_1 = sp.Symbol('Rx_1')
    Rx_2 = sp.Symbol('Rx_2')
    Rx_3 = sp.Symbol('Rx_3')
    Rx_4 = sp.Symbol('Rx_4')

    #Ry_1 = 0
    Ry_1 = sp.Symbol('Ry_1')
    Ry_2 = sp.Symbol('Ry_2')
    Ry_3 = sp.Symbol('Ry_3')
    Ry_4 = sp.Symbol('Ry_4')


    # Create the symbolic rotation matrix
    A1 = symbolic_rotation_matrix(angle_deg_1)
    A2 = symbolic_rotation_matrix(angle_deg_2)
    A3 = symbolic_rotation_matrix(angle_deg_3)
    A4 = symbolic_rotation_matrix(angle_deg_4)

    # Equations

    #[Rx_1,Rx_2,Rx_3,Rx_4,Ry_1,Ry_2,Ry_3,Ry_4,angle_deg_2,angle_deg_2,angle_deg_3,angle_deg_4]

    eq_fija1= Rx_1
    eq_fija2= Ry_1
    eq_fija3= angle_deg_1
 
    equation_pr12_i = Rx_1 + sp.cos(sp.rad(angle_deg_1))*u1_c_pr12x - sp.sin(sp.rad(angle_deg_1))*u1_c_pr12y -Rx_2-sp.cos(sp.rad(angle_deg_2))*u2_c_pr12x+sp.sin(sp.rad(angle_deg_2))*u2_c_pr12y
    equation_pr23_i = Rx_2 + sp.cos(sp.rad(angle_deg_2))*u2_c_pr23x - sp.sin(sp.rad(angle_deg_2))*u2_c_pr23y -Rx_3-sp.cos(sp.rad(angle_deg_3))*u3_c_pr23x+sp.sin(sp.rad(angle_deg_3))*u3_c_pr23y
    equation_pr34_i = Rx_3 + sp.cos(sp.rad(angle_deg_3))*u3_c_pr34x - sp.sin(sp.rad(angle_deg_3))*u3_c_pr34y -Rx_4-sp.cos(sp.rad(angle_deg_4))*u4_c_pr34x+sp.sin(sp.rad(angle_deg_4))*u4_c_pr34y


    equation_pr12_j = Ry_1 + sp.sin(sp.rad(angle_deg_1))*u1_c_pr12x + sp.sin(sp.rad(angle_deg_1))*u1_c_pr12y -Ry_2-sp.sin(sp.rad(angle_deg_2))*u2_c_pr12x-sp.cos(sp.rad(angle_deg_2))*u2_c_pr12y
    equation_pr23_j = Ry_2 + sp.sin(sp.rad(angle_deg_2))*u2_c_pr23x + sp.cos(sp.rad(angle_deg_2))*u2_c_pr23y -Ry_3-sp.sin(sp.rad(angle_deg_3))*u3_c_pr23x-sp.cos(sp.rad(angle_deg_3))*u3_c_pr23y
    equation_pr34_j = Ry_3 + sp.sin(sp.rad(angle_deg_3))*u3_c_pr34x + sp.cos(sp.rad(angle_deg_3))*u3_c_pr34y -Ry_4-sp.sin(sp.rad(angle_deg_4))*u4_c_pr34x-sp.cos(sp.rad(angle_deg_4))*u4_c_pr34y

    equation_ppr_a = angle_deg_3-angle_deg_4 - alfa
    equation_ppr_b = Ry_4-offset #Rx_4 - offset #Ry_4 

    C_q = [eq_fija1,eq_fija2,eq_fija3,
           equation_pr12_i,equation_pr12_j,
            equation_pr23_i,equation_pr23_j,
            equation_pr34_i,equation_pr34_j,
            equation_ppr_a,equation_ppr_b]

    # Use fsolve to find the solution with the initial estimates
    initial_estimates = [0,0,0, 0, 0, 0, 0, 0, 1, 1, 0]

    f = sp.lambdify([(Rx_1,Ry_1,angle_deg_1,Rx_2,Ry_2,Rx_3,Ry_3,angle_deg_3,Rx_4,Ry_4,angle_deg_4)], C_q)
    solution = fsolve(f, initial_estimates)

    return solution


def create_animation(l2, l3,offset):
    

    # Create a subplot with play button
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter'}]])

    # Set the range for axes
    # fig.update_xaxes(range=[-10, 15], constrain="domain", scaleanchor="y")
    # fig.update_yaxes(range=[-10, 15], scaleanchor="x")
    max_range = l2 + l3 +1
    fig.update_xaxes(range=[-max_range, max_range], constrain="domain", scaleanchor="y")
    fig.update_yaxes(range=[-l2*1.1, l2*1.1], scaleanchor="x")

    # Animation settings
    animation_settings = dict(frame=dict(duration=100, redraw=True), fromcurrent=True)

    # Add frames for the animation
    frames = []

    for angle_deg_2 in range(0, 361, 10):
        solution = slider_crank_position(l2, l3,offset, angle_deg_2)
        #Rx_2, Ry_2, Rx_3, Ry_3, Rx_4, Ry_4 = solution
        Rx_1, Ry_1 = solution[0], solution[1]
        angle_deg_1 = solution[2]

        Rx_2, Ry_2 = solution[3], solution[4]

        Rx_3, Ry_3 = solution[5], solution[6]
        angle_deg_3 = solution[7]

        Rx_4, Ry_4 = solution[8], solution[9]
        angle_deg_4 = solution[10]
        
        frame = go.Frame(data=[
            go.Scatter(x=[Rx_2, Rx_3], y=[Ry_2, Ry_3], mode='markers+lines', name='Crank'),
            go.Scatter(x=[Rx_3, Rx_4], y=[Ry_3, Ry_4], mode='markers+lines', name='Connecting Rod')
        ])
        
        frames.append(frame)

    fig.frames = frames

    # Set initial data
    solution = slider_crank_position(l2, l3,offset, 0)
    #Rx_2, Ry_2, Rx_3, Ry_3, Rx_4, Ry_4 = solution
    Rx_1, Ry_1 = solution[0], solution[1]
    angle_deg_1 = solution[2]

    Rx_2, Ry_2 = solution[3], solution[4]

    Rx_3, Ry_3 = solution[5], solution[6]
    angle_deg_3 = solution[7]

    Rx_4, Ry_4 = solution[8], solution[9]
    angle_deg_4 = solution[10]

    fig.add_trace(go.Scatter(x=[Rx_2, Rx_3], y=[Ry_2, Ry_3], mode='markers+lines', name='Crank'))
    fig.add_trace(go.Scatter(x=[Rx_3, Rx_4], y=[Ry_3, Ry_4], mode='markers+lines', name='Connecting Rod'))

    #add a title to the animation (centered)
    fig.update_layout(title_text="Slider-Crank Mechanism Animation", title_x=0.5)

    #add the name of the axis
    fig.update_xaxes(title_text="X")
    fig.update_yaxes(title_text="Y")

    # Set the layout for the animation
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, animation_settings])])])

    # Show the animation
    #fig.show()
    return fig




app = dash.Dash(__name__, external_stylesheets=['https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])
#app = JupyterDash(__name__, external_stylesheets=['https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

app.layout = html.Div([
    html.H1("Slider and Crank", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin': '20px 0'}), 
    html.H2("Analyzing the Piston Position", style={'textAlign': 'center', 'fontWeight': 'bold', 'margin': '20px 0'}),    
    html.Label("Crank:"),
    dcc.Input(id="l2-input", type="number", value=5),
    html.Label("Connecting Rod:"),
    dcc.Input(id="l3-input", type="number", value=7),
    html.Label("offset:"),
    dcc.Input(id="offset-input", type="number", value=1),
    html.Button("Update", id="update-button"),
    html.Div(id="range-info"),
    dcc.Graph(id="animation-graph"),
    dcc.Graph(id="piston-graph"),
    html.Div([
            html.H4("About"),
            html.A("My Blog - FOSSEngineer ", href="https://fossengineer.com", target="_blank"),
            html.Br(),
            html.A("Source Code", href="https://github.com/JAlcocerT/Slider-Crank", target="_blank"),
            #html.Br(),
            #html.Img(src='/assets/FOSSEngineer.png', style={'height': '50px', 'width': 'auto'}),  # Adjust height and width as needed
        ], style={'float': 'right'}) #style={'position': 'fixed', 'bottom': '0px', 'right': '85px'})
])

@app.callback(
    [Output("piston-graph", "figure"), Output("range-info", "children"), Output("animation-graph", "figure")],
    [Input("update-button", "n_clicks")],
    [dash.dependencies.State("l2-input", "value"),
     dash.dependencies.State("l3-input", "value"),
     dash.dependencies.State("offset-input", "value")]
)
def update_graph(n_clicks, l2, l3, offset):
    # Update piston graph
    angle_deg_2_values = list(range(0, 361, 10))
    Rx_4_values = []

    for angle_deg_2 in angle_deg_2_values:
        solution = slider_crank_position(l2, l3, offset, angle_deg_2)
        Rx_4 = solution[8]
        Rx_4_values.append(Rx_4)

    fig = go.Figure(go.Scatter(x=angle_deg_2_values, y=Rx_4_values, mode='lines+markers'))
    fig.update_layout(title="Piston Position Evolution vs Crankshaft Angle", title_x=0.5,
                      xaxis_title="Crankshaft Angle (Degrees)",
                      yaxis_title="Piston Position")

    range_of_motion = max(Rx_4_values) - min(Rx_4_values)
    #min_value = min(Rx_4_values)
    min_index = np.argmin(Rx_4_values)
    min_angle = angle_deg_2_values[min_index]

    
    range_info = f"Range of Motion: {range_of_motion:.2f}, Intake/Combustion: {min_angle:.2f}°, Compression/Exhaust:{360-min_angle:.2f}°"

    # Update animation graph
    animation_fig = create_animation(l2, l3, offset)

    return fig, range_info, animation_fig

# Start of the application
if __name__ == '__main__':
    
    app.run_server(debug=False, host="0.0.0.0", port=8050)