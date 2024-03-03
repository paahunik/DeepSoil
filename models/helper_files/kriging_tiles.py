import numpy as np
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)



def check_multiple_variogram_model(x,y,phi):
    for method in ["linear", "power", "gaussian", "spherical", "exponential", "hole-effect"]:
        OK = OrdinaryKriging(x, y, phi, variogram_model=method,
                             verbose=True, enable_plotting=False)

        print(OK.variogram_model_parameters)
        gridx = np.arange(-300, 300, 10, dtype="float64")
        gridy = np.arange(0, 600, 10, dtype="float64")
        zstar, ss = OK.execute("grid", xpoints=gridx, ypoints=gridy)  # n_closest_point?

        # print(zstar.shape, ss.shape)
        # print(gridx.shape, gridy.shape)
        # print(np.min(gridx), np.max(gridx))
        # print(np.min(gridy), np.max(gridy))

        # Plotting kriging predictions
        cax2 = plt.imshow(zstar, extent=(-300, 300, 0, 600), origin='lower')
        plt.scatter(x, y, c='k', marker='.')
        plt.colorbar(cax2)
        plt.title("Porosity estimate (" + method + ")")
        plt.savefig("estimate_" + method + ".png")
        plt.close()

        print(np.min(np.sqrt(ss)), np.max(np.sqrt(ss)))

        # plotting uncertainties
        cax = plt.imshow(np.sqrt(ss), extent=(-300, 300, 0, 600), origin="lower", vmin=0)
        plt.scatter(x, y, c="k", marker=".")
        plt.colorbar(cax)
        plt.title("Porosity Standard Deviation (" + method + ")")
        plt.savefig("Porosity_standard_deviation_" + method + ".png")
        plt.close()


if __name__ == '__main__':
    x = np.array([-100, 200, -290, 23, 101, 110])
    y = np.array([56, 100, 590, 470, 200, 25])
    phi = np.array([29.3, 21, 19.2, 29.1, 21.9, 28])


    target_x = 150
    target_y = 150

    distances = np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2)
    weights = 1.0 / distances * phi
    OK = OrdinaryKriging(x, y, weights, variogram_model='gaussian',
                         verbose=True, enable_plotting=False)

    print(OK)
    z_interp, sigma = OK.execute('grid', [target_x], [target_y])
    print("Interpolated value:", z_interp[0])
    print("Standard deviation:", sigma[0])









