import matplotlib.pyplot as plt
import matplotlib.widgets as w

fig = plt.figure(figsize=(15, 10))
SL, SW = 0.02, 0.14
ax_sl = fig.add_axes([SL + 0.05, 0.43, SW - 0.06, 0.04])
slider = w.Slider(ax_sl, 'Brush Size', 0, 20, valinit=0)
fig.savefig('test.png')
