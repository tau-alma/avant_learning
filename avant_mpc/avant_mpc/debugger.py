import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.widgets import Slider

goal = [0., 0., 0.]
model = torch.load("critic").eval()

inputs = torch.empty([40*40, 13])
i = 0
for x in range(-20, 20):
    for y in range(-20, 20):
        hdg = 0
        inputs[i] = torch.tensor([
            x, y, np.sin(goal[hdg]), np.cos(goal[hdg]), 
            goal[0], goal[1], np.sin(goal[2]), np.cos(goal[2]),
            0, 0, 0,
            0, 0
        ])
        i += 1

inputs = inputs.cuda()
q_vals = -model(inputs)
q_vals_np = q_vals.cpu().detach().numpy().reshape(40, 40).T

def update(val):
    hdg = slider.val
    for i, (x, y) in enumerate(np.ndindex(x_range.shape[0], y_range.shape[0])):
        inputs[i, 2:4] = torch.tensor([np.sin(hdg), np.cos(hdg)])
    
    inputs_cuda = inputs.cuda()
    q_vals = -model(inputs_cuda)
    q_vals_np = q_vals.cpu().detach().numpy().reshape(40, 40).T
    heatmap.set_data(q_vals_np)
    fig.canvas.draw_idle()

# Initial hdg value
initial_hdg = 0

# X and Y ranges
x_range = np.linspace(-20, 20, 40)
y_range = np.linspace(-20, 20, 40)

# Create the figure and the axis for the heatmap
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
heatmap = ax.matshow(q_vals_np, cmap='viridis')
ax.set_xticklabels([''] + list(range(-20, 21, 5)))
ax.set_yticklabels([''] + list(range(-20, 21, 5)))

# Create the slider
axcolor = 'lightgoldenrodyellow'
ax_hdg = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax=ax_hdg, label='HDG', valmin=-np.pi, valmax=np.pi, valinit=initial_hdg)

# Update the heatmap when the slider is changed
slider.on_changed(update)

# Show the plot
plt.show()
