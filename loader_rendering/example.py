import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from renderer import LoaderRenderer

ar = LoaderRenderer(1080, 5, store_frames=True)

# Draw one frame:
frame_static = ar.render_frame(np.array([0, 0, 0, 0]), np.array([2, 2, np.pi]), horizon=[], obstacles=[], comparision=[])

# Save a video clip and return the frames:
for i in range(100):
    if i % 2:
        ar.render_frame(np.array([3, -3, -np.pi/2, -0.1]), np.array([2, 2, np.pi]))
    else:
        ar.render_frame(np.array([-3, 3, np.pi/2, 0.1]), np.array([2, 2, np.pi]))
frames = ar.render_video("test.mp4")

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