# %%
import numpy as np
from ipycanvas import Canvas, hold_canvas
from ipywidgets import HBox, IntProgress, Play, VBox, link
from IPython.display import display
from sidecar import Sidecar
from PIL import Image
import colorcet as cc
from matplotlib.cm import get_cmap
import matplotlib as mpl
import time

# %%
"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate flow past cylinder
for an isothermal fluid

"""


# %%

# %%
def scale_wrapper(data):
    # min = np.min(data)
    # max = np.max(data)
    max = 0.08
    min = -max
    def scale(value):
        if value == np.nan:
            return 0
        scaled_value = (value - min) / (max - min)
        # return 255 if value > max else scaled_value * 255
        return 1.0 if value > max else scaled_value
    return scale



# %%
mymap = mpl.colormaps['cet_CET_D8']
def imgfromarray(data: np.array) -> np.array:
    return mymap(data, bytes=True)



# %%
sc = Sidecar(title='Sidecar Output')
scale = 1
# canvas = Canvas(width=scale * Nx, height=scale * Ny)
canvas = Canvas(width=800, height=800)
with sc:
    display(canvas)


# %%

# %%
def exit_nicely(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyboardInterrupt:
            print('User exited')
    return wrapper


# %%
# Simulation Main Loop

@exit_nicely
def run_sim():
    """Lattice Boltzmann Simulation"""
    
    # Simulation parameters
    Nx = 100  # resolution x-dir
    Ny = 30  # resolution y-dir
    rho0 = 100  # average density
    tau = 0.75 # 0.6  # collision timescale
    Nt = 5000  # number of timesteps
    plotRealTime = True  # switch on for plotting as the simulation goes along
    
    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array(
        [4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36]
    )  # sums to 1
    
    # Initial Conditions
    F = np.ones((Ny, Nx, NL))  # * rho0 / NL
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho
    
    # Cylinder boundary
    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = (X - Nx / 4) ** 2 + (Y - Ny / 2) ** 2 < (Ny / 4) ** 2

    with sc:
        sc.clear_output()
        canvas = Canvas(width=800, height=800)
        display(canvas)

    for it in range(Nt):
        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Set reflective boundaries
        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculate fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = (
                rho
                * w
                * (
                    1
                    + 3 * (cx * ux + cy * uy)
                    + 9 * (cx * ux + cy * uy) ** 2 / 2
                    - 3 * (ux**2 + uy**2) / 2
                )
            )

        F += -(1.0 / tau) * (F - Feq)

        # Apply boundary
        F[cylinder, :] = bndryF

        # plot in real time - color 1/2 particles blue, other half red
        if (plotRealTime and (it % 4) == 0) or (it == Nt - 1):
            ux[cylinder] = 0
            uy[cylinder] = 0
            vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)
            )
            vorticity[cylinder] = np.nan
            vorticity = np.ma.array(vorticity, mask=cylinder)
            vecscale = np.vectorize(scale_wrapper(vorticity))
            imgdata = imgfromarray(vecscale(vorticity))

            img = Image.fromarray(imgdata)
            img = img.resize((np.ones(2) * 6 * img.size).astype(np.int64))

            time.sleep(0.05)

            with sc:
                canvas.put_image_data(np.array(img))
        display((it, np.min(vorticity), np.max(vorticity)), clear=True)
    # Save figure


# %%
run_sim()

# %%
(np.ones(2)*2*img.size).astype(np.int64)
