import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from renderer import LoaderRenderer

lr = LoaderRenderer(1080, 5, store_frames=True)

# Draw one frame:
state = np.array([0, 0, 0, 0])
goal = np.array([2, 2, np.pi])
frame_static = lr.render_frame(state, goal, horizon=[], obstacles=[], comparision=[])

# Save a video clip and return the frames:
# (store_frames = True -> they are stored internally for compiling a .mp4 video clip afterwards)
for i in range(100):
    if i % 2:
        lr.render_frame(np.array([3, -3, -np.pi/2, -0.1]), np.array([2, 2, np.pi]))
    else:
        lr.render_frame(np.array([-3, 3, np.pi/2, 0.1]), np.array([2, 2, np.pi]))
frames = lr.render_video("test.mp4")

# Show result:
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 5.4))
im1 = ax1.imshow(frame_static)
im2 = ax2.imshow(frames[0])
ax1.set_title("Static Rendered Frame")
ax2.set_title("Video Playback")
ax1.axis("off")
ax2.axis("off")
def update(i):
    im2.set_data(frames[i])
    return [im2]
ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
plt.tight_layout()
plt.show()