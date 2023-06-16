import imageio

frames = []
for i in range(100):
    frames.append(imageio.imread(f"Part1/plots/spatial_plot/plot{i}.png"))

imageio.mimsave('./example.gif',
                frames,
                duration=100)
