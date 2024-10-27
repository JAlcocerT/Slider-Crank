#streamlit run slidercrankv1.py


import streamlit as st
import numpy as np
import plotly.graph_objects as go




import sympy as sp

from sympy import Eq, solve, pi
from sympy import lambdify
from scipy.optimize import fsolve


from sympy import symbols, Eq, cos, sin, diff, Matrix
from sympy import Matrix, Eq



# from sympy import symbols, Eq, cos, sin, diff, Matrix
# from sympy import Matrix, Eq


from sympy import Matrix, cos, sin, symbols

def rotation_matrix_2d(theta):
    """Generate a 2D rotation matrix for a given angle."""
    return Matrix([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

def generate_constraint_equations_pr(theta_i, theta_j, r_x_i, r_y_i, r_x_j, r_y_j, u_x_i, u_y_i, u_x_j, u_y_j):
    """Generate the position restraint equations for the PR."""
    R_i = Matrix([r_x_i, r_y_i])
    R_j = Matrix([r_x_j, r_y_j])
    local_vector_i = Matrix([u_x_i, u_y_i])
    local_vector_j = Matrix([u_x_j, u_y_j])
    
    transformed_vector_i = rotation_matrix_2d(theta_i) * local_vector_i
    transformed_vector_j = rotation_matrix_2d(theta_j) * local_vector_j

    constraint_equations = R_i + transformed_vector_i - R_j - transformed_vector_j
    return constraint_equations



def generate_equations_included_topology_pr(theta_i, theta_j, r_x_i, r_y_i, r_x_j, r_y_j, u_x_a, u_y_a, u_x_b, u_y_b):

    u_x_i, u_y_i, u_x_j, u_y_j = symbols('u_x_i u_y_i u_x_j u_y_j')


    # Generate the constraint equations
    equations = generate_constraint_equations_pr(theta_i, theta_j, r_x_i, r_y_i, r_x_j, r_y_j, u_x_i, u_y_i, u_x_j, u_y_j)
    
    # Substitute the symbolic variables with the provided values
    equations_substituted = equations.subs({u_x_i: u_x_a, u_y_i: u_y_a, u_x_j: u_x_b, u_y_j: u_y_b})
    
    
    # Return the equation and the symbols
    #return (equations)
    return equations_substituted

from sympy import symbols, Eq, tan, Matrix

def generate_constraint_equations_ppl(theta_i, theta_j, r_x_i, r_y_i, r_x_j, r_y_j, alfa, offset):
    """Generate the position restraint equations for the PPL."""

    expr1 = theta_j - theta_i - alfa
    expr2 = r_y_j - tan(alfa) * r_x_j - offset
    
    return Matrix([[expr1], [expr2]])


def four_bars_slidercrank_example_symbolic_full():

    theta_1, theta_2, theta_3, theta_4 = symbols('theta_1 theta_2 theta_3 theta_4')
    r_x_1, r_y_1, r_x_2, r_y_2, r_x_3, r_y_3, r_x_4, r_y_4 = symbols('r_x_1 r_y_1 r_x_2 r_y_2 r_x_3 r_y_3 r_x_4 r_y_4')

    L2, L3, L4, alfa, offset = symbols('L2 L3 L4 alfa, offset')


    u_x_1_12 = 0
    u_y_1_12 = 0
    u_x_2_12 = 0
    u_y_2_12 = 0
    restrictions_pr12 = generate_equations_included_topology_pr(theta_1, theta_2, r_x_1, r_y_1, r_x_2, r_y_2, u_x_1_12, u_y_1_12, u_x_2_12, u_y_2_12)

    u_x_2_23 = L2
    u_y_2_23 = 0
    u_x_3_23 = 0
    u_y_3_23 = 0
    restrictions_pr23 = generate_equations_included_topology_pr(theta_2, theta_3, r_x_2, r_y_2, r_x_3, r_y_3, u_x_2_23, u_y_2_23, u_x_3_23, u_y_3_23)

    u_x_3_34 = L3
    u_y_3_34 = 0
    u_x_4_34 = 0
    u_y_4_34 = 0
    restrictions_pr34 = generate_equations_included_topology_pr(theta_3, theta_4, r_x_3, r_y_3, r_x_4, r_y_4, u_x_3_34, u_y_3_34, u_x_4_34, u_y_4_34)
    
    restrictions_ppl14 = generate_constraint_equations_ppl(theta_1, theta_4, r_x_1, r_y_1, r_x_4, r_y_4, alfa, offset)
    
    # Stacking the matrices vertically to form a single large matrix
    restrictions_to_solve = restrictions_pr12.col_join(restrictions_pr23).col_join(restrictions_pr34).col_join(restrictions_ppl14)

    # Substituting the known values into the equations
    restrictions_substituted = restrictions_to_solve.subs({
        r_x_1: 0,
        r_y_1: 0,
        theta_1: 0 })

    # Create a list of equations from the matrix
    equation_list = [Eq(restrictions_substituted[i, 0], 0) for i in range(restrictions_substituted.shape[0])]


    return(equation_list)


def convert_floats(solution_dict):
    """Convert all values in the dictionary to float."""
    return {key: float(value.evalf()) if isinstance(value, (sp.Number, sp.Symbol)) else value 
            for key, value in solution_dict.items()}


def plot_slidercrank_position_json(new_input):
    # Extract values from the input dictionary using string keys
    r_x_2_value = new_input['r_x_2']
    r_y_2_value = new_input['r_y_2']
    r_x_3_value = new_input['r_x_3']
    r_y_3_value = new_input['r_y_3']
    r_x_4_value = new_input['r_x_4']
    r_y_4_value = new_input['r_y_4']

    #print(r_x_2_value)
    
    # Create traces for each line
    trace1 = go.Scatter(x=[r_x_2_value, r_x_3_value], y=[r_y_2_value, r_y_3_value], mode='lines+markers', name='Link 2-3')
    trace2 = go.Scatter(x=[r_x_3_value, r_x_4_value], y=[r_y_3_value, r_y_4_value], mode='lines+markers', name='Link 3-4')
    #trace3 = go.Scatter(x=[r_x_4_value, Separation], y=[r_y_4_value, 0], mode='lines+markers', name='Link 4-Base')
    
    # Define layout with equal aspect ratio and specified axis ranges
    layout = go.Layout(title='Four Bars Mechanism Visualization',
                       xaxis=dict(title='X Coordinate', scaleanchor="y", scaleratio=1, range=[-15, 15]),
                       yaxis=dict(title='Y Coordinate', scaleanchor="x", scaleratio=1, range=[-15, 15]),
                       showlegend=True,
                       autosize=False,  # Disable automatic sizing to maintain aspect ratio
                       width=500,  # Set width of the plot
                       height=500)  # Set height of the plot
    
    # Create figure and plot
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    fig.show()



def solve_and_plot_slidercrank(theta_2_value, L2_value, L3_value, offset_value, alfa_value):

        
    # Solve the equations
    #solutions = sp.solve( [eq.subs({theta_2: np.pi/2, L2: 8, L3: 8, offset: 0, alfa: 0}) for eq in four_bars_slidercrank_example_symbolic_full()] )
    solutions = sp.solve( [eq.subs({theta_2: theta_2_value, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value}) for eq in four_bars_slidercrank_example_symbolic_full()] )

    solutions_theta_3_1 = solutions[1]

    solution_json = {str(key): value for key, value in solutions_theta_3_1.items()}

    solution_json_converted = convert_floats(solution_json)
    print(solution_json_converted)
    plot_slidercrank_position_json(solution_json_converted)




# def plot_slidercrank_with_slider(theta_2_range, L2_value, L3_value, offset_value, alfa_value):

#     theta_2, L2, L3, offset, alfa = sp.symbols('theta_2 L2 L3 offset alfa')


#     # Generate solutions for a range of theta_2 values
#     solutions_list = []
#     print(theta_2_range)
#     for theta_2_loop in theta_2_range:
#         #print(theta_2_loop)
#         solutions = sp.solve( [eq.subs({theta_2: theta_2_loop, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value}) for eq in four_bars_slidercrank_example_symbolic_full()] )
#         #print(solutions)
#         selected_solution = solutions[1]

        
#         solution_json = {str(key): value for key, value in selected_solution.items()}
#         solution_json_converted = convert_floats(solution_json)
#         solutions_list.append(solution_json_converted)

#      # Generate traces for the plot
#     traces = []
#     for solution in solutions_list:
#         trace1 = go.Scatter(x=[solution['r_x_2'], solution['r_x_3']],
#                             y=[solution['r_y_2'], solution['r_y_3']],
#                             mode='lines+markers', name='Link 2-3')
#         trace2 = go.Scatter(x=[solution['r_x_3'], solution['r_x_4']],
#                             y=[solution['r_y_3'], solution['r_y_4']],
#                             mode='lines+markers', name='Link 3-4')
#         traces.extend([trace1, trace2])

#     # Create the slider steps
#     steps = []
#     for i, theta_2 in enumerate(theta_2_range):
#         step = dict(args=[{"visible": [False] * len(traces)}],  # Set all traces to invisible
#                     method="restyle",
#                     label=f"{theta_2:.2f}")
#         step["args"][0]["visible"][i*2:i*2+2] = [True, True]  # Toggle i'th trace to "visible"
#         steps.append(step)

#     sliders = [dict(active=10, yanchor="top", steps=steps)]

#     # Define layout
#     layout = go.Layout(title='Four Bars Mechanism Visualization',
#                        xaxis=dict(title='X Coordinate', scaleanchor="y", scaleratio=1, range=[-15, 15]),
#                        yaxis=dict(title='Y Coordinate', scaleanchor="x", scaleratio=1, range=[-15, 15]),
#                        showlegend=True,
#                        autosize=False,
#                        width=600,
#                        height=600,
#                        sliders=sliders)

#     fig = go.Figure(data=traces, layout=layout)
#     #fig.show()
#     return(fig)




# def plot_theta2_vs_rx4(theta_2_range, L2_value, L3_value, offset_value, alfa_value):
#     # Generate solutions for a range of theta_2 values
#     #solutions_list = []
    
#     theta_2, r_x_4 = symbols('theta_2, r_x_4')
#     L2, L3, alfa, offset = symbols('L2 L3 alfa, offset')

#     theta_2_values = []
#     r_x_4_values = []
#     for theta_2_loop in theta_2_range:
#         solutions = sp.solve( [eq.subs({theta_2: theta_2_loop, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value}) for eq in four_bars_slidercrank_example_symbolic_full()] )
        
#         selected_solution = solutions[1]

#         r_x_4_values.append(float(selected_solution[r_x_4]))
#         theta_2_values.append(theta_2_loop)


#     # Extract theta_2 and r_x_4 values for plotting
#     # theta_2_values = [theta_2 for theta_2 in theta_2_range]
#     # r_x_4_values = [solution['r_x_4'] for solution in solutions_list]

#     # Create trace
#     trace = go.Scatter(x=theta_2_values, y=r_x_4_values, mode='lines+markers', name='Theta_2 vs r_x_4')

#     # Define layout
#     layout = go.Layout(title='Theta_2 vs r_x_4 Plot',
#                        xaxis=dict(title='Theta_2'),
#                        yaxis=dict(title='r_x_4'),
#                        showlegend=True,
#                        autosize=False,
#                        width=600,
#                        height=600)

#     fig = go.Figure(data=[trace], layout=layout)
#     #fig.show()
#     return(fig)




def jacobian_slider_crank(): #ADAPT IT TO CONSIDER ANY MOVEMENT CONSTRAINT!!!

    #     # Define the symbols
    theta_2, theta_3, theta_4 = symbols('theta_2 theta_3 theta_4')
    r_x_2, r_y_2, r_x_3, r_y_3, r_x_4, r_y_4 = symbols('r_x_2 r_y_2 r_x_3 r_y_3 r_x_4 r_y_4')

    # Extract left-hand side of each equation
    exprs = [eq.lhs for eq in four_bars_slidercrank_example_symbolic_full()]

    # Add the additional constraint
    exprs.append(theta_2)

    # Convert the list to a Matrix
    f = Matrix(exprs)


    # Variables with respect to which we differentiate
    variables = [r_x_2, r_y_2, theta_2, r_x_3, r_y_3, theta_3, r_x_4, r_y_4, theta_4]

    # Compute the Jacobian
    jacobian_extended = f.jacobian(variables)
    return(jacobian_extended)



def C_t_slider_crank(): #ADAPT IT TO CONSIDER ANY MOVEMENT CONSTRAINT!!!

    #     # Define the symbols
    theta_2, theta_3, theta_4 ,time= symbols('theta_2 theta_3 theta_4 time')
    r_x_2, r_y_2, r_x_3, r_y_3, r_x_4, r_y_4 = symbols('r_x_2 r_y_2 r_x_3 r_y_3 r_x_4 r_y_4')

    # Extract left-hand side of each equation
    exprs = [eq.lhs for eq in four_bars_slidercrank_example_symbolic_full()]

    # Add the additional constraint
    exprs.append(theta_2-np.pi/4*time)

    # Convert the list to a Matrix
    f = Matrix(exprs)


    # Variables with respect to which we differentiate
    variables = [time]

    # Compute the Jacobian
    C_t = f.jacobian(variables)
    return(C_t)


# def plot_theta2_vs_vx4_w_cte(theta_2_range, L2_value, L3_value, offset_value, alfa_value):
#     # Generate solutions for a range of theta_2 values
#     #solutions_list = []
    
#     theta_2, theta_3 = symbols('theta_2, theta_3')
#     L2, L3, alfa, offset = symbols('L2 L3 alfa, offset')

#     C_q_inv = jacobian_slider_crank().inv() #dependent on q
#     C_t = C_t_slider_crank() #not dependant of q neither t here, constant


#     theta_2_values = []
#     v_x_4_values = []
#     for theta_2_loop in theta_2_range:

#         #solve the position
#         solutions = sp.solve( [eq.subs({theta_2: theta_2_loop, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value}) for eq in four_bars_slidercrank_example_symbolic_full()] )
        
#         q = solutions[1]

#         #now calculate q_dot
#         q_dot = -C_q_inv*C_t
        

#         q_dot = q_dot.subs({theta_2: theta_2_loop, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value, theta_3: q[theta_3]})
#         #print(q_dot)
#         v_x_4_values.append(float(q_dot[6]))

        
    
#         theta_2_values.append(theta_2_loop)
   

#     # Create trace
#     trace = go.Scatter(x=theta_2_values, y=v_x_4_values, mode='lines+markers', name='Theta_2 vs v_x_4 ????')

#     # Define layout
#     layout = go.Layout(title='Theta_2 vs v_x_4 Plot',
#                        xaxis=dict(title='Theta_2'),
#                        yaxis=dict(title='r_x_4'),
#                        showlegend=True,
#                        autosize=False,
#                        width=600,
#                        height=600)

#     fig = go.Figure(data=[trace], layout=layout)
#     #fig.show()
#     return(fig)


# import sympy as sp
# import plotly.graph_objs as go

def generate_solutions(theta_2_range, L2_value, L3_value, offset_value, alfa_value):
    theta_2, L2, L3, offset, alfa = sp.symbols('theta_2 L2 L3 offset alfa')

    solutions_list = []
    for theta_2_loop in theta_2_range:
        solutions = sp.solve([eq.subs({theta_2: theta_2_loop, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value}) for eq in four_bars_slidercrank_example_symbolic_full()])
        selected_solution = solutions[1]
        solution_json = {str(key): value for key, value in selected_solution.items()}
        solution_json_converted = convert_floats(solution_json)  # Assuming you have this function defined elsewhere
        solutions_list.append(solution_json_converted)
    
    return solutions_list


import sympy as sp
import plotly.graph_objs as go
import math  # for radians and degrees conversion

def generate_solutions(theta_2_range, L2_value, L3_value, offset_value, alfa_value):
    theta_2, L2, L3, offset, alfa = sp.symbols('theta_2 L2 L3 offset alfa')

    solutions_list = []
    for theta_2_rad in theta_2_range:
        solutions = sp.solve([eq.subs({theta_2: theta_2_rad, L2: L2_value, L3: L3_value, offset: offset_value, alfa: alfa_value}) for eq in four_bars_slidercrank_example_symbolic_full()])
        selected_solution = solutions[1]
        solution_json = {str(key): value for key, value in selected_solution.items()}
        solution_json_converted = convert_floats(solution_json)  # Assuming you have this function defined elsewhere
        solutions_list.append(solution_json_converted)
    
    return solutions_list

def combined_plot_functions(theta_2_range, L2_value, L3_value, offset_value, alfa_value):
    solutions_list = generate_solutions(theta_2_range, L2_value, L3_value, offset_value, alfa_value)

    # For the first plot
    traces = []
    for solution in solutions_list:
        trace1 = go.Scatter(x=[solution['r_x_2'], solution['r_x_3']],
                            y=[solution['r_y_2'], solution['r_y_3']],
                            mode='lines+markers', name='Link 2-3')
        trace2 = go.Scatter(x=[solution['r_x_3'], solution['r_x_4']],
                            y=[solution['r_y_3'], solution['r_y_4']],
                            mode='lines+markers', name='Link 3-4')
        traces.extend([trace1, trace2])

    # Slider setup for the first plot
    steps = []
    for i, theta_2_rad in enumerate(theta_2_range):
        theta_2_deg = math.degrees(theta_2_rad)  # Convert radians to degrees for the slider labels
        step = dict(args=[{"visible": [False] * len(traces)}],
                    method="restyle",
                    label=f"{theta_2_deg:.2f}")  # Displayed in degrees
        step["args"][0]["visible"][i*2:i*2+2] = [True, True]
        steps.append(step)

    sliders = [dict(active=10, yanchor="top", steps=steps)]
    layout1 = go.Layout(title='Four Bars Mechanism Visualization',
                        xaxis=dict(title='Theta_2 (degrees)', scaleanchor="y", scaleratio=1, range=[-15, 15]),
                        yaxis=dict(title='Y Coordinate', scaleanchor="x", scaleratio=1, range=[-15, 15]),
                        showlegend=True, autosize=False, width=600, height=600, sliders=sliders)
    fig1 = go.Figure(data=traces, layout=layout1)

    # For the second plot
    theta_2_values_deg = [math.degrees(theta_2_rad) for theta_2_rad in theta_2_range]  # Convert radians to degrees for plotting
    r_x_4_values = [solution['r_x_4'] for solution in solutions_list]
    trace = go.Scatter(x=theta_2_values_deg, y=r_x_4_values, mode='lines+markers', name='Theta_2 vs r_x_4')
    layout2 = go.Layout(title='Theta_2 vs r_x_4 Plot',
                        xaxis=dict(title='Theta_2 (degrees)'),
                        yaxis=dict(title='r_x_4'),
                        showlegend=True, autosize=False, width=600, height=600)
    fig2 = go.Figure(data=[trace], layout=layout2)

    return fig1, fig2

# # Usage:
# fig1, fig2 = combined_plot_functions(theta_2_range, L2_value, L3_value, offset_value, alfa_value)
# fig1.show()
# fig2.show()








st.title("Simulation App")

# Slider to select L2, L3, offset, and alfa values
L2 = st.slider('L2 Value', 0.0, 10.0, 1.0)
L3 = st.slider('L3 Value', 0.0, 10.0, 2.0)
offset = st.slider('Offset Value', 0.0, 1.0, 0.0)
alfa = st.slider('Alfa Value', -5, 5, 0)

if st.button('Simulate'):
    theta_2_range = np.linspace(0, 2*np.pi, 100)
    
    # Call your functions and plot
    # fig3 = plot_theta2_vs_vx4_w_cte(theta_2_range, L2, L3, offset, alfa)
    # fig2 = plot_theta2_vs_rx4(theta_2_range, L2, L3, offset, alfa)
    # fig1 = plot_slidercrank_with_slider(theta_2_range, L2, L3, offset, alfa)
    fig1, fig2 = combined_plot_functions(theta_2_range, L2, L3, offset, alfa)
    # Display plots
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    #st.plotly_chart(fig3)