import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt


# Function to read and plot the signal from a text file
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        # Extract the file name from the path
        selected_file = file_path.split("/")[-1]  # For Unix-like paths
        # selected_file = file_path.split("\\")[-1]  # For Windows paths
        selected_file_label["text"] = selected_file
        data = read_signal_data(file_path)
        if data is not None:
            plot_signal(data)


# Function to read the signal data from the text file
def read_signal_data(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

        if len(lines) < 3:
            print("Invalid file format. The file should contain at least 3 lines.")
            return None

        num_samples = int(lines[2])
        data = []

        for i in range(3, 3 + num_samples):
            parts = lines[i].split()
            if len(parts) != 2:
                print("Invalid line format.")
                return None
            index, amplitude = float(parts[0]), float(parts[1])
            data.append((index, amplitude))

        return data

    except Exception as e:
        print("Error reading the file:", e)
        return None


# Function to plot the signal (both continuous and discrete)
def plot_signal(data):
    if len(data) == 0:
        print("No data to plot.")
        return

    index = [sample[0] for sample in data]
    amplitude = [sample[1] for sample in data]

    plt.figure(figsize=(10, 4))

    # Continuous representation
    plt.subplot(1, 2, 1)
    plt.plot(index, amplitude, label="Continuous Signal", marker='o')
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Continuous Signal Representation")
    plt.legend()

    # Discrete representation
    plt.subplot(1, 2, 2)
    plt.stem(index, amplitude, linefmt='-b', markerfmt='ob', basefmt=' ', label="Discrete Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Discrete Signal Representation")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


# Function to generate and plot a sinusoidal or cosinusoidal signal
def generate_signal():
    plt.figure(figsize=(8, 4))

    signal_type = signal_type_combobox.get()
    A = float(A_entry.get())
    analog_frequency = float(analog_frequency_entry.get())
    sampling_frequency = float(sampling_frequency_entry.get())
    phase_shift = float(phase_shift_entry.get())

    t = np.arange(0, 1, 1 / sampling_frequency)  # Time vector

    if signal_type == "sin":
        signal = A * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
        signal_label = "Sinusoidal Signal"
    elif signal_type == "cos":
        signal = A * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
        signal_label = "Cosinusoidal Signal"
    else:
        print("Invalid signal type.")
        return

    plt.plot(t, signal, label=signal_label)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Generated Signal")
    plt.legend()
    plt.grid(True)
    plt.show()


# Create the main window
root = tk.Tk()
root.geometry("800x500")
root.title("DSP Task 1")

header_label = ttk.Label(root, text="DSP Task 1", font=("Arial", 20))
header_label.pack(pady=20)

upload_button = ttk.Button(root, text="Upload Text File", command=browse_file)
upload_button.pack()

# Label to display the selected file name
selected_file_label = ttk.Label(root)
selected_file_label.pack(pady=10)

# Create a combo box to select signal type
signal_type_combobox = ttk.Combobox(root, values=("sin", "cos"))
signal_type_combobox.set("sin")
signal_type_combobox.pack(pady=10)

# Create entry fields for user input
A_label = ttk.Label(root, text="A:")
A_label.pack()
A_entry = ttk.Entry(root)
A_entry.insert(0, "")
A_entry.pack()

analog_frequency_label = ttk.Label(root, text="Analog Frequency:")
analog_frequency_label.pack()
analog_frequency_entry = ttk.Entry(root)
analog_frequency_entry.insert(0, "")
analog_frequency_entry.pack()

sampling_frequency_label = ttk.Label(root, text="Sampling Frequency:")
sampling_frequency_label.pack()
sampling_frequency_entry = ttk.Entry(root)
sampling_frequency_entry.insert(0, "")
sampling_frequency_entry.pack()

phase_shift_label = ttk.Label(root, text="Phase Shift:")
phase_shift_label.pack()
phase_shift_entry = ttk.Entry(root)
phase_shift_entry.insert(0, "")
phase_shift_entry.pack()

# Create a button to generate the signal
generate_button = ttk.Button(root, text="Generate Signal", command=generate_signal)
generate_button.pack(pady=20)

root.mainloop()
