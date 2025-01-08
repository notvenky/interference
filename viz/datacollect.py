import argparse
import time
import h5py
import csv
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from reskin_sensor import ReSkinProcess

if __name__ == "__main__":

    logs_directory = "logs"
    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory)

    current_date_str = datetime.now().strftime("%Y_%m_%d")

    day_directory = os.path.join(logs_directory, current_date_str)
    if not os.path.exists(day_directory):
        os.makedirs(day_directory)

    run_directory = os.path.join(day_directory, datetime.now().strftime("test_run_%H_%M_%S"))
    os.makedirs(run_directory)

    parser = argparse.ArgumentParser(
        description="Test code to run a ReSkin streaming process in the background. Allows data to be collected without code blocking"
    )
    # fmt: off
    parser.add_argument("-p", "--port", type=str, help="port to which the microcontroller is connected", required=True,)
    parser.add_argument("-b", "--baudrate", type=str, help="baudrate at which the microcontroller is streaming data", default=115200,)
    parser.add_argument("-n", "--num_mags", type=int, help="number of magnetometers on the sensor board", default=5,)
    parser.add_argument("-tf", "--temp_filtered", action="store_true", help="flag to filter temperature from sensor output",)
    # parser.add_argument("-tg", "--target", type=str, help="target position (e.g., tl for top_left)", required=True, choices=["tl", "tr", "fl", "fr", "il", "ir", "bl", "br"])
    # parser.add_argument("-t", "--time", type=int, help="amount of time to buffer data for", required=True, default=10,)

    # fmt: on
    args = parser.parse_args()

    target_mapping = {
        "tl": "top_left",
        "tr": "top_right",
        "fl": "front_left",
        "fr": "front_right",
        "il": "inward_left",
        "ir": "inward_right",
        "bl": "back_left",
        "br": "back_right",
    }

    # run_directory_name = f"{target_mapping[args.target]}_run_{datetime.now().strftime('%H_%M_%S')}"
    # run_directory = os.path.join(day_directory, run_directory_name)
    # os.makedirs(run_directory)

    os.chdir(run_directory)

    # Create sensor stream
    sensor_stream = ReSkinProcess(
        num_mags=args.num_mags,
        port=args.port,
        baudrate=args.baudrate,
        burst_mode=True,
        device_id=1,
        temp_filtered=True,
    )

    # Start sensor stream
    sensor_stream.start()
    time.sleep(0.1)

    # Buffer data for two seconds and return buffer
if sensor_stream.is_alive():
    sensor_stream.start_buffering()
    buffer_start = time.time()
    time.sleep(3)

    sensor_stream.pause_buffering()
    buffer_stop = time.time()

    # Get buffered data
    buffered_data = sensor_stream.get_buffer()

    if buffered_data is not None:
        print(
            "Time elapsed: {}, Number of datapoints: {}".format(
                buffer_stop - buffer_start, len(buffered_data)
            )
        )

        # # Subtraction of values using the first Bx, By, Bz readings
        # first_reading = [float(i) for i in buffered_data[0].data]
        # for reading in buffered_data:
        #     for i in range(len(reading.data)):
        #         if i != 0 and i != 4:  # Skip T0 and T1 which are at indices 0 and 4
        #             reading.data[i] = str(float(reading.data[i]) - first_reading[i])

        # Save as h5 dataset
        data_to_save = [list(map(float, reading.data)) for reading in buffered_data]

        with h5py.File('buffered_data.h5', 'w') as hf:
            hf.create_dataset("data", data=data_to_save)

        # Save as CSV
        with open('buffered_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in buffered_data:
                writer.writerow(row.data)

        # Save as JSON
        with open('buffered_data.json', 'w') as jsonfile:
            json.dump([row.data for row in buffered_data], jsonfile)

        # Get a specified number of samples
        test_samples = sensor_stream.get_data(num_samples=10)
        print(
            "Columns: ",
            ", \t".join(
                [
                    ((not args.temp_filtered)*"T{0}, \t" + "Bx{0}, \tBy{0}, \tBz{0}").format(ind)
                    for ind in range(args.num_mags)
                ]
            ),
        )
        for sid, sample in enumerate(test_samples):
            print(
                "Sample {}: ".format(sid + 1)
                + str(["{:.2f}".format(d) for d in sample.data])
            )

        # # Print post-subtraction values
        # print("Post-Subtraction Values:")
        # for sid, sample in enumerate(buffered_data):
        #     print(
        #         "Sample {}: ".format(sid + 1)
        #         + ", ".join(["{:.2f}".format(float(d)) for d in sample.data[2:]])
        #     )

        # Pause sensor stream
        bx0_values = [sample[0] for sample in data_to_save]
        by0_values = [sample[1] for sample in data_to_save]
        bz0_values = [sample[2] for sample in data_to_save]
        bx1_values = [sample[3] for sample in data_to_save]
        by1_values = [sample[4] for sample in data_to_save]
        bz1_values = [sample[5] for sample in data_to_save]

        # Calculate standard deviations
        std_bx0 = np.std(bx0_values)
        std_by0 = np.std(by0_values)
        std_bz0 = np.std(bz0_values)
        std_bx1 = np.std(bx1_values)
        std_by1 = np.std(by1_values)
        std_bz1 = np.std(bz1_values)

        # Plot the standard deviations for each dimension
        labels = ["bx0", "by0", "bz0", "bx1", "by1", "bz1"]
        std_values = [std_bx0, std_by0, std_bz0, std_bx1, std_by1, std_bz1]

        plt.figure(figsize=(10, 6))  # Optional: Define figure size
        plt.bar(labels, std_values)
        plt.ylabel('Standard Deviation')
        plt.title('Standard Deviations for Each Dimension')
        plt.savefig('standard_deviations.png')  # Save the standard deviations plot
        plt.close()  # Close the plot

        # Plot the CSV data and save it as well
        with open('buffered_data.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
            data = np.array(data, dtype=float)  # Convert string data to float
            
        for col_index, label in enumerate(labels):
            plt.plot(data[:, col_index], label=label)  # Plotting each column in the csv file
            
        plt.xlabel('Sample Number')
        plt.ylabel('Value')
        plt.title('CSV Data Plot')
        plt.legend()
        plt.savefig('csv_data_plot.png')  # Save the CSV data plot
        plt.close()  # Close the plot

        sensor_stream.pause_streaming()

        sensor_stream.join()

        os.chdir("..")