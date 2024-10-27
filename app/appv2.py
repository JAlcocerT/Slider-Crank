import streamlit as st
import numpy as np
from sympy import symbols, sin, cos, lambdify, asin, pi, Abs
import plotly.graph_objects as go

def plot_graph(l2, l3):
    alfa = symbols('alfa')
    
    # Define beta in terms of alfa using the absolute value for positive results
    beta_expr = Abs(asin(sin(alfa)/2)) + pi

    # Define s in terms of alfa using the given equations
    s_expr = l2 * cos(alfa) + l3 * cos(beta_expr)
    
    # Derivatives
    ds_dalfa_symbolic = s_expr.diff(alfa)
    d2s_dalfa2_symbolic = ds_dalfa_symbolic.diff(alfa)
    
    # Convert symbolic expressions to functions
    s_function = lambdify(alfa, s_expr, "numpy")
    ds_dalfa_function = lambdify(alfa, ds_dalfa_symbolic, "numpy")
    d2s_dalfa2_function = lambdify(alfa, d2s_dalfa2_symbolic, "numpy")
    
    # Generate alfa values and evaluate functions
    alfa_values_symbolic = np.linspace(0, 2*np.pi, 1000)
    s_values_symbolic = s_function(alfa_values_symbolic)
    ds_dalfa_values = ds_dalfa_function(alfa_values_symbolic)
    d2s_dalfa2_values = d2s_dalfa2_function(alfa_values_symbolic)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=alfa_values_symbolic, y=s_values_symbolic, mode='lines', name='s(alfa)'))
    fig.add_trace(go.Scatter(x=alfa_values_symbolic, y=ds_dalfa_values, mode='lines', name="ds/dalfa"))
    fig.add_trace(go.Scatter(x=alfa_values_symbolic, y=d2s_dalfa2_values, mode='lines', name="d^2s/dalfa^2"))
    
    # Layout
    fig.update_layout(
        title="s and its derivatives as functions of alfa",
        xaxis_title="alfa (radians)",
        yaxis_title="Value",
        legend_title="Legend",
        font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple")
    )
    
    st.plotly_chart(fig)


st.title("Crank Mechanism Analysis")

# User input
l2 = st.number_input("Enter the value for l2:", value=4.0)
l3 = st.number_input("Enter the value for l3:", value=8.0)

# Button and display logic
if st.button("Generate Plot"):
    plot_graph(l2, l3)


# import streamlit as st
# import numpy as np
# from sympy import symbols, sin, cos, lambdify, asin, pi
# import plotly.graph_objects as go

# def plot_graph(l2, l3):
#     alfa = symbols('alfa')
    
#     # Define beta in terms of alfa
#     beta_expr = asin(sin(alfa)/2) + pi

#     # Define s in terms of alfa using the given equations
#     s_expr = l2 * cos(alfa) + l3 * cos(beta_expr)
    
#     # Derivatives
#     ds_dalfa_symbolic = s_expr.diff(alfa)
#     d2s_dalfa2_symbolic = ds_dalfa_symbolic.diff(alfa)
    
#     # Convert symbolic expressions to functions
#     s_function = lambdify(alfa, s_expr, "numpy")
#     ds_dalfa_function = lambdify(alfa, ds_dalfa_symbolic, "numpy")
#     d2s_dalfa2_function = lambdify(alfa, d2s_dalfa2_symbolic, "numpy")
    
#     # Generate alfa values and evaluate functions
#     alfa_values_symbolic = np.linspace(0, 2*np.pi, 1000)
#     s_values_symbolic = s_function(alfa_values_symbolic)
#     ds_dalfa_values = ds_dalfa_function(alfa_values_symbolic)
#     d2s_dalfa2_values = d2s_dalfa2_function(alfa_values_symbolic)

#     # Plot
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=alfa_values_symbolic, y=s_values_symbolic, mode='lines', name='s(alfa)'))
#     fig.add_trace(go.Scatter(x=alfa_values_symbolic, y=ds_dalfa_values, mode='lines', name="ds/dalfa"))
#     fig.add_trace(go.Scatter(x=alfa_values_symbolic, y=d2s_dalfa2_values, mode='lines', name="d^2s/dalfa^2"))
    
#     # Layout
#     fig.update_layout(
#         title="s and its derivatives as functions of alfa",
#         xaxis_title="alfa (radians)",
#         yaxis_title="Value",
#         legend_title="Legend",
#         font=dict(family="Courier New, monospace", size=12, color="RebeccaPurple")
#     )
    
#     st.plotly_chart(fig)


# st.title("Crank Mechanism Analysis")

# # User input
# l2 = st.number_input("Enter the value for l2:", value=4.0)
# l3 = st.number_input("Enter the value for l3:", value=8.0)

# # Button and display logic
# if st.button("Generate Plot"):
#     plot_graph(l2, l3)