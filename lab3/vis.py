import plotly.graph_objects as go

# Create random data with numpy
import numpy as np

N = 100
x0 = [5, 10, 15, 19, 20, 25]
x1 = [5, 10, 15, 19]
y0 = [0.0629, 0.1347, 0.3416, 2.877, 5.66051, 175]
y1 = [0.0228,      0.01788,      0.01972,      0.2330]
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=x0, y=y0,
                         mode='lines+markers',
                         name='unified memory'))
fig.add_trace(go.Scatter(x=x1, y=y1,
                         mode='lines+markers',
                         name='device memory'))

fig.show()
