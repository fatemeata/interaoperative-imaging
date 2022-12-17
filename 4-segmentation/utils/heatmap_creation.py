import numpy as np
import matplotlib.pyplot as plt


class Heatmap:
    def __init__(self):
        pass

    # 2D Gaussian function
    def twoD_Gaussian(self, batch, xo, yo, sigma):
        x, y = batch
        a = 1. / (2 * sigma ** 2)
        g = np.exp(-(a * (x - xo) ** 2 + a * (y - yo) ** 2))
        return g

    def apply_gauss(self, x0, y0, sigma, image_shape):
        w, h = image_shape, image_shape
        self.y, self.x = np.mgrid[0:h, 0:w]
        gauss = self.twoD_Gaussian((self.x, self.y), x0, y0, sigma)
        # gauss = self.two_d_gauss(image_shape, x0, y0, sigma, trunc=15)
        gauss = gauss.reshape((image_shape, image_shape))
        return gauss

    def transparent_cmap(self, cmap, N=255):
        """Copy colormap and set alpha values"""
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap

    def display_heatmap(self, image_tensor, start_gauss, end_gauss):

        mycmap = self.transparent_cmap(plt.cm.Reds)
        image = image_tensor.permute(1, 2, 0)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, cmap='gray')
        s_cb = ax.contourf(self.x, self.y, start_gauss, 15, cmap=mycmap)
        e_cb = ax.contourf(self.x, self.y, end_gauss, 15, cmap=mycmap)
        plt.colorbar(s_cb, ax=None)
        plt.colorbar(e_cb)
        plt.show()

    def line_heatmap(self, line_x, line_y, trunc, sigma, img_shape):
        xo, yo = line_x
        x1, y1 = line_y
        line_heatmap = np.zeros((img_shape, img_shape))
        xx, yy = np.meshgrid(range(line_heatmap.shape[0]), range(line_heatmap.shape[1]))

        # Fit a line to the points line_x and line_y
        line_coefficients = np.polyfit(line_x, line_y, deg=1)
        line_values = np.polyval(line_coefficients, xx)
        # Calculate the distance of each point to the line
        dist = np.abs(line_values - yy)
        # Set the value of points with distance less than the truncation value to the value of the Gaussian function
        line_heatmap[dist < trunc] = np.exp(-dist[dist < trunc] ** 2 / (2 * sigma ** 2))

        line_heatmap[:int(x1), :int(y1)] = 0
        line_heatmap[int(xo):, int(yo):] = 0

        return line_heatmap

    def display_heatmap_line(self, image_tensor, line_gauss):
        mycmap = self.transparent_cmap(plt.cm.Reds)
        image = image_tensor.permute(1, 2, 0)

        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, cmap='gray')
        s_cb = ax.contourf(self.x, self.y, line_gauss, 15, cmap=mycmap)
        plt.colorbar(s_cb)
        plt.show()


