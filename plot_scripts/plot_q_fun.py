import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import torch

# Load the model
model = torch.load("avant_critic").cuda().eval()

# Define a function to update the heatmap
def update(val):
    hdg = slider.val
    for i, (x, y) in enumerate(np.ndindex(len(x_range), len(y_range))):
        inputs[i, :] = torch.tensor([
            x_range[x], y_range[y],    
            np.sin(hdg), np.cos(hdg), 
            np.sin(0), np.cos(0), 
            0, 0, 0, 
            0, 0
        ])
    
    inputs_cuda = inputs.cuda()
    with torch.no_grad():
        q_vals = model(inputs_cuda)**2
    q_vals_np = q_vals.cpu().numpy().reshape(len(y_range), len(x_range))
    ax.clear()
    ax.plot_surface(X, Y, q_vals_np, cmap='viridis')
    ax.set_title('Q-values Heatmap')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_zlabel('Q-values')
    fig.canvas.draw_idle()

# Initial hdg value
initial_hdg = 0

# X and Y ranges
x_range = np.linspace(-5, 5, 100)
y_range = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_range, y_range)

# Prepare inputs tensor
inputs = torch.empty([len(x_range) * len(y_range), 11])  # Adjust based on your model's input

# Create the figure and the 3D axis for the plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Initial plot
for i, (x, y) in enumerate(np.ndindex(len(x_range), len(y_range))):
    inputs[i, :] = torch.tensor([
        x_range[x], y_range[y],    
        np.sin(initial_hdg),  np.cos(initial_hdg), 
        np.sin(0), np.cos(0), 
        0, 0, 0, 
        0, 0
    ])

inputs_cuda = inputs.cuda()
with torch.no_grad():
    q_vals = model(inputs_cuda)**2  # Initial computation
q_vals_np = q_vals.cpu().numpy().reshape(len(y_range), len(x_range))
ax.plot_surface(X, Y, q_vals_np, cmap='viridis')

# Create the slider
axcolor = 'lightgoldenrodyellow'
ax_hdg = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax=ax_hdg, label='HDG', valmin=-np.pi, valmax=np.pi, valinit=initial_hdg)

# Update the heatmap when the slider is changed
slider.on_changed(update)

# Set the title and labels
ax.set_title('Q-values Heatmap')
ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_zlabel('Q-values')

# Show the plot
plt.show()
