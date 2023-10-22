import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt


def browse_file(button_number):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        # Extract the file name from the path
        selected_file = file_path.split("/")[-1]
        selected_files[button_number]["text"] = selected_file  # Display the selected file name
        data = read_signal_data(file_path)
        if button_number == 1:
            data1.extend(data)
        else:
            data2.extend(data)


def read_signal_data(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

        if len(lines) < 3:
            print("Invalid file format.")
            return None

        signal_type = int(lines[0])
        is_periodic = int(lines[1])
        num_samples = int(lines[2])
        data = []

        for i in range(3, 3 + num_samples):
            parts = lines[i].split()
            if signal_type == 0:  # Time domain
                if len(parts) != 2:
                    print("Invalid line format in time domain signal.")
                    return None
                index, amplitude = float(parts[0]), float(parts[1])
            elif signal_type == 1:  # Frequency domain
                if len(parts) != 3:
                    print("Invalid line format in frequency domain signal.")
                    return None
                index, amplitude, phase_shift = float(parts[0]), float(parts[1]), float(parts[2])
            else:
                print("Invalid signal type.")
                return None

            data.append((index, amplitude))

        return data

    except Exception as e:
        print("Error reading the file:", e)
        return None


def add_signals(data1, data2):
    if len(data1) != len(data2):
        print("Error: The signals have different lengths.")
        return None

    summed_signal = [(sample1[0], sample1[1] + sample2[1]) for sample1, sample2 in zip(data1, data2)]
    return summed_signal


def sub_signals(data1, data2):
    if len(data1) != len(data2):
        print("Error: The signals have different lengths.")
        return None

    subtracted_signal = [(sample1[0], sample1[1] - sample2[1]) for sample1, sample2 in zip(data1, data2)]
    return subtracted_signal


def multiply_signal(data, constant):
    if constant == -1:
        # Invert the signal
        multiplied_signal = [(sample[0], -sample[1]) for sample in data]
    else:
        # Multiply the signal by the constant
        multiplied_signal = [(sample[0], sample[1] * constant) for sample in data]
    return multiplied_signal

def square_signal(data):
    squared_signal = [(sample[0], sample[1] ** 2) for sample in data]
    return squared_signal

def shift_signal(data, constant):
    shifted_signal = [(sample[0] + constant, sample[1]) for sample in data]
    return shifted_signal

def normalize_signal(data, normalize_type):
    if normalize_type == 0:  # Normalize to [0, 1]
        max_amplitude = max(sample[1] for sample in data)
        min_amplitude = min(sample[1] for sample in data)

        if max_amplitude == min_amplitude:
            return data  # Avoid division by zero

        normalized_signal = [(sample[0], (sample[1] - min_amplitude) / (max_amplitude - min_amplitude)) for sample in data]
    elif normalize_type == 1:  # Normalize to [-1, 1]
        max_amplitude = max(sample[1] for sample in data)
        min_amplitude = min(sample[1] for sample in data)

        if max_amplitude == min_amplitude:
            return data  # Avoid division by zero

        normalized_signal = [(sample[0], 2 * (sample[1] - min_amplitude) / (max_amplitude - min_amplitude) - 1) for sample in data]
    else:
        return data

    return normalized_signal



def accumulate_signal(data):
    accumulated_signal = []
    accumulated_amplitude = 0

    for sample in data:
        accumulated_amplitude += sample[1]
        accumulated_signal.append((sample[0], accumulated_amplitude))

    return accumulated_signal

def plot_signal(data):
    if len(data) == 0:
        print("No data to plot.")
        return

    index = [sample[0] for sample in data]
    amplitude = [sample[1] for sample in data]

    plt.figure(figsize=(10, 5))

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


# main window
root = tk.Tk()
root.geometry("800x500")
root.title("DSP Task 2")

header_label = ttk.Label(root, text="DSP Task 2", font=("Arial", 20))
header_label.pack(pady=20)

selected_files = {}  # Dictionary to store the selected file labels

data1 = []  # List to store data for signal 1
data2 = []  # List to store data for signal 2

upload_button1 = ttk.Button(root, text="Upload Text File for Signal 1", command=lambda: browse_file(1))
upload_button1.pack()

selected_file_label1 = ttk.Label(root, text="Selected File 1: ")
selected_file_label1.pack(pady=10)
selected_files[1] = selected_file_label1  # Store the label in the dictionary

upload_button2 = ttk.Button(root, text="Upload Text File for Signal 2", command=lambda: browse_file(2))
upload_button2.pack()

selected_file_label2 = ttk.Label(root, text="Selected File 2: ")
selected_file_label2.pack(pady=10)
selected_files[2] = selected_file_label2  # Store the label in the dictionary

add_button = ttk.Button(root, text="Add Signals", command=lambda: plot_signal(add_signals(data1, data2)))
add_button.pack()

sub_button = ttk.Button(root, text="Subtract Signals", command=lambda: plot_signal(sub_signals(data1, data2)))
sub_button.pack()

const_label = ttk.Label(root, text="Enter A Constant value to use it in Shifting or Multiplying :")
const_label.pack()

constant_value = tk.DoubleVar(value=1.0)
constant_entry = ttk.Entry(root, textvariable=constant_value, width=10)
constant_entry.pack()

mul_button = ttk.Button(root, text="Multiply Signal by Constant",command=lambda: plot_signal(multiply_signal(data1, constant_value.get())))
mul_button.pack()

shift_button = ttk.Button(root, text="Shift Signal", command=lambda: plot_signal(shift_signal(data1, constant_value.get())))
shift_button.pack()

squaring_button = ttk.Button(root, text="Squaring Signal",command=lambda: plot_signal(square_signal(data1)))
squaring_button.pack()

normalize_type_var = tk.IntVar(value=0)  # Default: Normalize to [0, 1]

normalize_label = ttk.Label(root, text="Normalization Type:")
normalize_label.pack()

normalize_radiobutton0 = ttk.Radiobutton(root, text="0 to 1", variable=normalize_type_var, value=0)
normalize_radiobutton1 = ttk.Radiobutton(root, text="-1 to 1", variable=normalize_type_var, value=1)

normalize_radiobutton0.pack()
normalize_radiobutton1.pack()

normalize_button = ttk.Button(root, text="Normalize Signal", command=lambda: plot_signal(normalize_signal(data1, normalize_type_var.get())))
normalize_button.pack()


accumulate_button = ttk.Button(root, text="Accumulate Signal", command=lambda: plot_signal(accumulate_signal(data1)))
accumulate_button.pack()


root.mainloop()
