import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt

# Global variables
data = None  # Store the loaded data
sampling_frequency = 1.0  # Default sampling frequency

def browse_file_1():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        selected_file = file_path.split("/")[-1]
        selected_file_label["text"] = selected_file
        data = read_signal_data_1(file_path)

def read_signal_data_1(file_path):
    try:
        with open(file_path, "r") as file:
            lines = file.read().splitlines()

        if len(lines) < 3:
            raise ValueError("Invalid file format.")

        num_samples = int(lines[2])
        data = []

        for i in range(3, 3 + num_samples):
            parts = lines[i].split()

            if len(parts) != 2:
                raise ValueError("Invalid line format.")

            index, amplitude = float(parts[0]), float(parts[1])
            data.append((index, amplitude))

        return data

    except Exception as e:
        print("Error reading the file:", e)
        return None

def calculate_fourier_transform(data, sampling_frequency):
    N = len(data)
    time = [point[0] for point in data]
    signal = [point[1] for point in data]

    freq = np.fft.fftfreq(N, 1.0 / sampling_frequency)
    fft_result = np.fft.fft(signal)

    amplitude = np.abs(fft_result) / N
    phase = np.angle(fft_result)

    return freq, amplitude, phase

def plot_frequency_amplitude_and_phase(freq, amplitude, phase):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(freq, amplitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Frequency vs Amplitude")
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(freq, phase)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.title("Frequency vs Phase")
    plt.grid()

    plt.tight_layout()
    plt.show()

def DFT_button_3():
    global sampling_frequency
    sampling_frequency = float(sampling_frequency_entry.get())
    if data is None:
        print("Please upload a file before generating the Fourier transform.")
        return
    freq, amplitude, phase = calculate_fourier_transform(data, sampling_frequency)
    plot_frequency_amplitude_and_phase(freq, amplitude, phase)

def update_signal_component():
    global data
    component_index = int(component_index_entry.get())
    new_amplitude = float(new_amplitude_entry.get())
    new_phase = float(new_phase_entry.get())
    
    if data is not None and 0 <= component_index < len(data):
        data[component_index] = (data[component_index][0], new_amplitude)
        DFT_button_3()  # Recalculate and plot the Fourier transform

def reconstruct_signal():
    global data
    sampling_frequency = float(sampling_frequency_entry.get())
    
    reconstructed_freq = float(reconstructed_freq_entry.get())
    reconstructed_amplitude = float(reconstructed_amplitude_entry.get())
    
    if data is not None:
        N = len(data)
        reconstructed_signal = np.zeros(N, dtype=complex)
        for i, (freq, amp) in enumerate(data):
            phase = 0.0  # Default phase
            if reconstructed_freq == freq:
                phase = np.pi * reconstructed_phase / 180.0  # Convert to radians
            reconstructed_signal[i] = np.exp(2j * np.pi * reconstructed_freq * i / sampling_frequency) * reconstructed_amplitude
        reconstructed_signal = np.fft.ifft(reconstructed_signal)
        plot_reconstructed_signal(np.real(reconstructed_signal))

def plot_reconstructed_signal(reconstructed_signal):
    plt.figure()
    plt.plot(reconstructed_signal)
    plt.title("Reconstructed Signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

root = tk.Tk()
root.geometry("800x700")
root.title("DSP Task")

header_label = ttk.Label(root, text="DSP Task", font=("Arial", 20))
header_label.pack(pady=20)

upload_button = ttk.Button(root, text="Upload Text File", command=browse_file_1)
upload_button.pack()

selected_file_label = ttk.Label(root)
selected_file_label.pack(pady=10)

sampling_frequency_label = ttk.Label(root, text="Sampling Frequency:")
sampling_frequency_label.pack()

sampling_frequency_entry = ttk.Entry(root)
sampling_frequency_entry.insert(0, "")
sampling_frequency_entry.pack()

generate_button = ttk.Button(root, text="Generate Signal", command=DFT_button_3)
generate_button.pack(pady=20)

# Components modification section
component_index_label = ttk.Label(root, text="Component Index:")
component_index_label.pack()

component_index_entry = ttk.Entry(root)
component_index_entry.insert(0, "0")
component_index_entry.pack()

new_amplitude_label = ttk.Label(root, text="New Amplitude:")
new_amplitude_label.pack()

new_amplitude_entry = ttk.Entry(root)
new_amplitude_entry.insert(0, "0.0")
new_amplitude_entry.pack()

new_phase_label = ttk.Label(root, text="New Phase (radians):")
new_phase_label.pack()

new_phase_entry = ttk.Entry(root)
new_phase_entry.insert(0, "0.0")
new_phase_entry.pack()

update_component_button = ttk.Button(root, text="Update Component", command=update_signal_component)
update_component_button.pack(pady=20)

# Reconstruction section
reconstructed_freq_label = ttk.Label(root, text="Reconstructed Frequency:")
reconstructed_freq_label.pack()

reconstructed_freq_entry = ttk.Entry(root)
reconstructed_freq_entry.insert(0, "0.0")
reconstructed_freq_entry.pack()

reconstructed_amplitude_label = ttk.Label(root, text="Reconstructed Amplitude:")
reconstructed_amplitude_label.pack()

reconstructed_amplitude_entry = ttk.Entry(root)
reconstructed_amplitude_entry.insert(0, "0.0")
reconstructed_amplitude_entry.pack()

reconstructed_phase_label = ttk.Label(root, text="Reconstructed Phase (degrees):")
reconstructed_phase_label.pack()

reconstructed_phase_entry = ttk.Entry(root)
reconstructed_phase_entry.insert(0, "0.0")
reconstructed_phase_entry.pack()

reconstruct_button = ttk.Button(root, text="Reconstruct Signal", command=reconstruct_signal)
reconstruct_button.pack(pady=20)

root.mainloop()
