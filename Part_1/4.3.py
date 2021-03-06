import numpy as np
import matplotlib.pyplot as plt

xdata = 7 * np.random.random(100)
ydata = np.sin(xdata) + 0.25 * np.random.random(100)
zdata = np.exp(xdata) + 0.25 * np.random.random(100)

fig = plt.figure(figsize=(9, 6))
# Create 3D container
ax = plt.axes(projection = '3d')
# Visualize 3D scatter plot
ax.scatter3D(xdata, ydata, zdata)
# Give labels
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# Save figure
plt.savefig('3d_scatter.png', dpi = 300, bbox_inches = 'tight',
    pad_inches = 0);
