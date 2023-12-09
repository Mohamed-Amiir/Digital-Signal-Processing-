import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_signal(ax, signal, title, color):
    ax.stem(signal, basefmt=color + '-', markerfmt=color + 'o', label=title)
    ax.legend()
    ax.set_title(title)

def perform_convolution():

    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
    input_data = file_content.split('\n')[3:]
    input_data = [line.split() for line in input_data if line.strip()]
    indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
    InputIndicesSignal1 = np.array(indices)
    signal1 = np.array(values)

    # InputIndicesSignal1 = [-2, -1, 0, 1]
    # InputSamplesSignal1 = [1, 2, 1, 1]
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
    input_data = file_content.split('\n')[3:]
    input_data = [line.split() for line in input_data if line.strip()]
    indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
    InputIndicesSignal2 = np.array(indices)
    signal2 = np.array(values)

    # InputIndicesSignal2 = [0, 1, 2, 3, 4, 5]
    # InputSamplesSignal2 = [1, -1, 0, 0, 1, 1]

    # Create two example signals
    # signal1 = np.array([1, 2, 3, 4])
    # signal2 = np.array([0.5, 1, 0.5])

    # Perform convolution in the frequency domain
    result_size = len(signal1) + len(signal2) - 1
    fft_signal1 = np.fft.fft(signal1, result_size)
    fft_signal2 = np.fft.fft(signal2, result_size)
    conv_freq = np.fft.ifft(fft_signal1 * fft_signal2)

    # Plot each signal in an independent window
    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    plot_signal(axs[0], signal1, 'Signal 1', 'b')
    plot_signal(axs[1], signal2, 'Signal 2', 'g')
    plot_signal(axs[2], np.real(conv_freq), 'Convolution (Frequency Domain)', 'm')
    plt.show()

def perform_correlation():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
    input_data = file_content.split('\n')[3:]
    input_data = [line.split() for line in input_data if line.strip()]
    indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
    InputIndicesSignal1 = np.array(indices)
    signal1 = np.array(values)

    # InputIndicesSignal1 = [-2, -1, 0, 1]
    # InputSamplesSignal1 = [1, 2, 1, 1]
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
    input_data = file_content.split('\n')[3:]
    input_data = [line.split() for line in input_data if line.strip()]
    indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
    InputIndicesSignal2 = np.array(indices)
    signal2 = np.array(values)

    # InputIndicesSignal2 = [0, 1, 2, 3, 4, 5]
    # InputSamplesSignal2 = [1, -1, 0, 0, 1, 1]

    # Create two example signals
    # signal1 = np.array([1, 2, 3, 4])
    # signal2 = np.array([0.5, 1, 0.5])l

    result_size = len(signal1) + len(signal2) - 1
    fft_signal1 = np.fft.fft(signal1, result_size)
    fft_signal2 = np.fft.fft(signal2, result_size)
    corr_freq = np.fft.ifft(fft_signal1.conjugate() * fft_signal2)

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    plot_signal(axs[0], signal1, 'Signal 1', 'b')
    plot_signal(axs[1], signal2, 'Signal 2', 'g')
    plot_signal(axs[2], np.real(corr_freq), 'Correlation (Frequency Domain)', 'm')
    plt.show()

# Create the main window
root = tk.Tk()
root.geometry("300x100")
root.title("Fast Convolution and Correlation GUI")

# Create buttons to perform convolution and correlation
button_convolution = ttk.Button(root, text="Perform Convolution", command=perform_convolution)
button_convolution.grid(row=2, column=0, pady=10, padx=5)

button_correlation = ttk.Button(root, text="Perform Correlation", command=perform_correlation)
button_correlation.grid(row=2, column=1, pady=10, padx=5)

# Start the Tkinter main loop
root.mainloop()
