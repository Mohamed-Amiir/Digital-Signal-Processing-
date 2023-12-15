import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import tkinter as tk
from tkinter import filedialog, messagebox

class FIRFilterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("FIR Filter Application")

        # Initialize variables
        self.filter_type_var = tk.StringVar(value='Low pass')
        self.fs_var = tk.StringVar(value='8000')
        self.stop_band_attenuation_var = tk.StringVar(value='50')
        self.fc_var = tk.StringVar(value='1500')
        self.transition_band_var = tk.StringVar(value='500')
        self.lp_coefficients = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Filter Type:").grid(row=0, column=0, padx=10, pady=5)
        filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        self.filter_type_menu = tk.OptionMenu(self.master, self.filter_type_var, *filter_types)
        self.filter_type_menu.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Sampling Frequency:").grid(row=1, column=0, padx=10, pady=5)
        self.fs_entry = tk.Entry(self.master, textvariable=self.fs_var, width=50)
        self.fs_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Stop Band Attenuation:").grid(row=2, column=0, padx=10, pady=5)
        self.stop_band_attenuation_entry = tk.Entry(self.master, textvariable=self.stop_band_attenuation_var, width=50)
        self.stop_band_attenuation_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Cutoff Frequency:").grid(row=3, column=0, padx=10, pady=5)
        self.fc_entry = tk.Entry(self.master, textvariable=self.fc_var, width=50)
        self.fc_entry.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Transition Band:").grid(row=4, column=0, padx=10, pady=5)
        self.transition_band_entry = tk.Entry(self.master, textvariable=self.transition_band_var, width=50)
        self.transition_band_entry.grid(row=4, column=1, padx=10, pady=5)

        # Button to run the FIR filter
        tk.Button(self.master, text="Run FIR Filter", command=self.run_fir_filter).grid(row=5, column=0, columnspan=2, pady=10)

    def run_fir_filter(self):
        # Get user input
        filter_type = self.filter_type_var.get()
        fs = float(self.fs_var.get())
        stop_band_attenuation = float(self.stop_band_attenuation_var.get())
        fc = float(self.fc_var.get())
        transition_band = float(self.transition_band_var.get())

        # Design the FIR filter
        num_taps = self.compute_num_taps(transition_band, stop_band_attenuation, fs)
        filter_coefficients = firwin(num_taps, fc, fs=fs)

        # Apply the filter (dummy data for input_signal, replace with your data)
        input_signal = np.random.randn(1000)  # Replace with your actual input data
        filtered_signal = lfilter(filter_coefficients, 1.0, input_signal)

        # Plot the results
        self.plot_results(input_signal, filtered_signal)

        # Save coefficients to a text file
        self.lp_coefficients = filter_coefficients
        self.save_coefficients(filter_coefficients)

    def compute_num_taps(self, delta_f, stop_attenuation, fs):
        delta_omega = 2 * np.pi * delta_f / fs
        num_taps = int(6.6 * fs / delta_omega)
        if num_taps % 2 == 0:  # Ensure it is odd
            num_taps += 1
        return num_taps

    def plot_results(self, input_signal, filtered_signal):
        plt.figure(figsize=(10, 6))
        plt.plot(input_signal, label='Input Signal', linewidth=2)
        plt.plot(filtered_signal, label='Filtered Signal', linewidth=2)
        plt.legend()
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('FIR Filtered Signal')
        plt.show()

    def save_coefficients(self, coefficients, filename='lpf_coefficients.txt'):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            np.savetxt(file_path, coefficients, delimiter=',')
            messagebox.showinfo("Saved", "LPF coefficients saved to {}".format(file_path))

def main():
    root = tk.Tk()
    app = FIRFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
