import os
import pandas as pd
import matplotlib.pyplot as plt
import h5py

base_folder = "/home/venky/temp/anyskin/viz/logs/2024_12_09"

run_folders = sorted(os.listdir(base_folder))
csv_files = [os.path.join(base_folder, folder, "buffered_data.csv") for folder in run_folders]

first_5_files = csv_files[:5]
last_5_files = csv_files[5:10]
last_file = csv_files[-1]

def extract_last_readings(files):
    readings = []
    for file in files:
        df = pd.read_csv(file)
        df.columns = [
            "Bx0", "By0", "Bz0", "Bx1", "By1", "Bz1",
            "Bx2", "By2", "Bz2", "Bx3", "By3", "Bz3",
            "Bx4", "By4", "Bz4"
        ]
        last_row = df.iloc[-1].values
        readings.append(last_row)
    return readings

first_5_readings = extract_last_readings(first_5_files)
last_5_readings = extract_last_readings(last_5_files)
last_reading = extract_last_readings([last_file])[0]  # Single reading, no list

b_labels = ["Bx0", "By0", "Bz0", "Bx1", "By1", "Bz1",
            "Bx2", "By2", "Bz2", "Bx3", "By3", "Bz3",
            "Bx4", "By4", "Bz4"]

# Plot "New Skins" with last run overlay
plt.figure(figsize=(10, 6))
for i, readings in enumerate(first_5_readings, start=1):
    plt.plot(b_labels, readings, marker='o', label=f"Run {i}")
plt.plot(b_labels, last_reading, linestyle='dotted', color='red', label="Last Run")
plt.title("New Skins with odd one out")
plt.xlabel("Sensors")
plt.ylabel("Magnitude")
plt.ylim(-2000, 500)
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.savefig("new_skins.png")
plt.close()

# Plot "Old Skins"
plt.figure(figsize=(10, 6))
for i, readings in enumerate(last_5_readings, start=6):
    plt.plot(b_labels, readings, marker='o', label=f"Run {i}")
plt.title("Old Skins")
plt.xlabel("Sensors")
plt.ylabel("Magnitude")
plt.ylim(-2000, 500)
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.savefig("old_skins_plot.png")
plt.close()

# Save the data to an HDF5 file
with h5py.File("all_readings.h5", "w") as h5_file:
    h5_file.create_dataset("new_skins", data=first_5_readings)
    h5_file.create_dataset("old_skins", data=last_5_readings)
    h5_file.create_dataset("last_run", data=last_reading)
