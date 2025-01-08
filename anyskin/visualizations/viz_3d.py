#!/usr/bin/env python

import time
import numpy as np
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import sys
import pygame
from datetime import datetime
from anyskin import AnySkinProcess
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize(port, file=None, viz_mode="3axis", scaling=7.0, record=False):
    if file is None:
        sensor_stream = AnySkinProcess(
            num_mags=5,
            port=port,
        )
        sensor_stream.start()
        time.sleep(1.0)
        filename = "data/data_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        load_data = np.loadtxt(file)
        data_len = 0

    # Sensor chip locations for 2D and 3D
    chip_locations = np.array(
        [
            [204, 222],  # center
            [130, 222],  # left
            [279, 222],  # right
            [204, 157],  # up
            [204, 290],  # down
        ]
    )
    chip_3d_locations = np.array(
        [
            [0, 0, 0],     # center
            [-10, 0, 0],   # left
            [10, 0, 0],    # right
            [0, 10, 0],    # up
            [0, -10, 0],   # down
        ]
    )
    chip_xy_rotations = np.array([-np.pi / 2, -np.pi / 2, np.pi, np.pi / 2, 0.0])

    # Initialize Pygame for 2D visualization
    pygame.init()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    bg_image_path = os.path.join(dir_path, "images/viz_bg.png")
    bg_image = pygame.image.load(bg_image_path)
    bg_image = pygame.transform.scale(bg_image, (400, int(400 * bg_image.get_size()[1] / bg_image.get_size()[0])))
    window = pygame.display.set_mode((400, int(400 * bg_image.get_size()[1] / bg_image.get_size()[0])), pygame.SRCALPHA)
    pygame.display.set_caption("Sensor Data Visualization")

    # Initialize Matplotlib for 3D visualization
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    plt.ion()  # Enable interactive mode for live plotting

    def get_baseline():
        """Calculate baseline values."""
        baseline_data = sensor_stream.get_data(num_samples=5)
        baseline_data = np.array(baseline_data)[:, 1:]
        return np.mean(baseline_data, axis=0)

    baseline = get_baseline() if file is None else np.zeros(15)
    running = True
    in_3d_mode = False
    data = []

    def draw_2d(data):
        """Draw the 2D visualization."""
        window.blit(bg_image, (0, 0))
        data = data.reshape(-1, 3)
        data_mag = np.linalg.norm(data, axis=1)
        for magid, chip_location in enumerate(chip_locations):
            # Draw circles for magnitude
            pygame.draw.circle(
                window, (255, 83, 72), chip_location, data_mag[magid] / scaling
            )
            # Draw arrows for direction
            rotation_mat = np.array(
                [
                    [np.cos(chip_xy_rotations[magid]), -np.sin(chip_xy_rotations[magid])],
                    [np.sin(chip_xy_rotations[magid]), np.cos(chip_xy_rotations[magid])],
                ]
            )
            data_xy = np.dot(rotation_mat, data[magid, :2])
            arrow_end = (
                chip_location[0] + data_xy[0] / scaling,
                chip_location[1] + data_xy[1] / scaling,
            )
            pygame.draw.line(window, (0, 255, 0), chip_location, arrow_end, 2)
        pygame.display.update()

    def draw_3d(data):
        """Update the 3D visualization."""
        ax.clear()
        ax.set_title("3D Magnetic Field Visualization")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")

        data = data.reshape(-1, 3)
        x, y, z = chip_3d_locations[:, 0], chip_3d_locations[:, 1], chip_3d_locations[:, 2]
        u, v, w = data[:, 0], data[:, 1], data[:, 2]

        # Plot sensor positions and magnetic field vectors
        ax.scatter(x, y, z, color="red", s=50, label="Sensor Positions")
        ax.quiver(x, y, z, u, v, w, length=5 / scaling, normalize=True, color="blue")
        ax.legend()
        plt.pause(0.01)

    while running:
        if in_3d_mode:
            # 3D visualization loop
            if file is None:
                sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
                sensor_data -= baseline
                data.append(sensor_data)
            draw_3d(sensor_data)
        else:
            # 2D visualization loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_3:
                        in_3d_mode = True
                        plt.show()  # Switch to 3D mode
                    elif event.key == pygame.K_b:
                        baseline = get_baseline() if file is None else np.zeros(15)

            if file is None:
                sensor_data = sensor_stream.get_data(num_samples=1)[0][1:]
                sensor_data -= baseline
                data.append(sensor_data)
            draw_2d(sensor_data)

    # Cleanup
    if file is None:
        sensor_stream.pause_streaming()
        sensor_stream.join()
    pygame.quit()
    plt.close(fig)
    if record and file is None:
        np.savetxt(f"{filename}.txt", np.array(data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize magnetic field data in 2D and 3D.")
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", default="/dev/cu.usbmodem101")
    parser.add_argument("-f", "--file", type=str, help="path to load data from", default=None)
    parser.add_argument("-v", "--viz_mode", type=str, help="visualization mode", default="3axis", choices=["magnitude", "3axis"])
    parser.add_argument("-s", "--scaling", type=float, help="scaling factor for visualization", default=7.0)
    parser.add_argument("-r", "--record", action="store_true", help="record data")
    args = parser.parse_args()
    visualize(args.port, args.file, args.viz_mode, args.scaling, args.record)
