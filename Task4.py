import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
import signalcompare as test
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import Label, Entry, Button, Text, Scrollbar, END

# Function to perform Fourier transform and plot the results
def plot_fourier_transform(samples, sampling_frequency):

    # Calculate the Fourier transform
    # Calculate the angular frequency array
    omega = np.fft.fftfreq(len(samples), d=1/sampling_frequency)

    # Calculate the Fourier transform
    fft_coefficients = np.fft.fft(samples)

    # Calculate magnitude and phase spectra
    magnitude_spectrum = np.abs(fft_coefficients)
    phase_spectrum = np.angle(fft_coefficients)

    # file_path = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 4\\Test Cases\\DFT\\Output_Signal_DFT_A,Phase.txt"
    # with open(file_path, 'r') as file:
    #     file_content = file.read() 
    # input_data = file_content.split('\n')[3:]
    # input_data = [line.split() for line in input_data if line.strip()]
    # magn, phas = zip(*[(float(index), float(value)) for index, value in input_data])
    # phas_output = np.array(phas)
    # magn_output = np.array(magn)
    # if (test.SignalComaprePhaseShift(phas_output, phase_spectrum) == True):
    #     print("ACCEPTED")
    # else:
    #     print("WRONG phase")

    # if (test.SignalComapreAmplitude(magn_output, magnitude_spectrum)==True):
    #     print("ACCEPTED")
    # else:
    #     print("WRONG magn")

    # Create a new window for the plots
    plot_window = tk.Toplevel(root)
    plot_window.title('Fourier Transform Results')

    # Plot the magnitude spectrum
    plt.subplot(2, 1, 1)
    plt.plot(omega, magnitude_spectrum)
    plt.title('Frequency vs Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    # Plot the phase spectrum
    plt.subplot(2, 1, 2)
    plt.plot(omega, phase_spectrum)
    plt.title('Frequency vs Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')

    plt.show()


# Function to handle the button click event
def analyze_signal(sampling_freq_entry):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                file_content = file.read()
        input_data = file_content.split('\n')[3:]
        input_data = [line.split() for line in input_data if line.strip()]
        indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
        signal_samples = np.array(values)
        # Get user input for the sampling frequency
        sampling_frequency = float(sampling_freq_entry.get())

        # Call the function to perform Fourier transform and plot the results
        plot_fourier_transform(signal_samples, sampling_frequency)
        


   
# def upload_file():
#             file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
#             if file_path:
#                 with open(file_path, 'r') as file:
#                     file_content = file.read()
#                 result_label.config(text="File uploaded successfully.")
#                 input_data = file_content.split('\n')[3:]
#                 input_data = [line.split() for line in input_data if line.strip()]
#                 indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
#                 signal_samples = np.array(values)
#                 return signal_samples
#             return 

# Create the main tkinter window
root = tk.Tk()
root.title('Signal Analyzer')

# Create and place GUI components
Label(root, text="Enter Sampling Frequency (Hz):").pack(pady=5)
sampling_freq_entry = Entry(root)
sampling_freq_entry.pack(pady=5)

analyze_button = Button(root, text="Fourior Trans DFT", command=lambda: analyze_signal(sampling_freq_entry))
analyze_button.pack(pady=10)

result_label = Label(root, text="")
result_label.pack()

# Start the tkinter event loop
root.mainloop()
