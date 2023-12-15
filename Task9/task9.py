import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import tkinter as tk
from tkinter import filedialog

class FIRFilterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("FIR Filter Application")

        # Initialize variables
        self.input_signal = np.array([])
        self.filter_type = tk.StringVar()
        self.cutoff_freq = tk.StringVar()
        self.num_taps = tk.StringVar()
        self.fs = tk.StringVar()

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Filter Type:").grid(row=0, column=0, padx=10, pady=5)
        filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        self.filter_type_menu = tk.OptionMenu(self.master, self.filter_type, *filter_types)
        self.filter_type.set(filter_types[0])
        self.filter_type_menu.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Sampling Frequency:").grid(row=1, column=0, padx=10, pady=5)
        self.fs_entry = tk.Entry(self.master, width=50)
        self.fs_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Stop Band Attenuation:").grid(row=2, column=0, padx=10, pady=5)
        self.cutoff_freq_entry = tk.Entry(self.master, width=50)
        self.cutoff_freq_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.master, text="FC:").grid(row=3, column=0, padx=10, pady=5)
        self.num_taps_entry = tk.Entry(self.master, width=50)
        self.num_taps_entry.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Transition Band:").grid(row=4, column=0, padx=10, pady=5)
        self.transition_band_entry = tk.Entry(self.master, width=50)
        self.transition_band_entry.grid(row=4, column=1, padx=10, pady=5)

        # Button to run the FIR filter
        tk.Button(self.master, text="Run FIR Filter", command=self.run_fir_filter).grid(row=5, column=0, columnspan=2, pady=10)

    def run_fir_filter(self):
        # Get user input
        self.input_signal = np.array(self.input_entry.get().split(','), dtype=float)
        filter_type = self.filter_type.get()
        cutoff_freq = [float(x) for x in self.cutoff_freq_entry.get().split(',')]
        num_taps = int(self.num_taps_entry.get())
        fs = float(self.fs_entry.get())

        # Design the FIR filter
        filter_coefficients = firwin(num_taps, [cutoff_freq[0], cutoff_freq[1]], pass_zero=False, fs=fs)

        # Apply the filter
        filtered_signal = lfilter(filter_coefficients, 1.0, self.input_signal)

        # Plot the results
        self.plot_results(self.input_signal, filtered_signal)

        # Save coefficients to a text file
        self.save_coefficients(filter_coefficients)

    def plot_results(self, input_signal, filtered_signal):
        plt.figure(figsize=(10, 6))
        plt.plot(input_signal, label='Input Signal', linewidth=2)
        plt.plot(filtered_signal, label='Filtered Signal', linewidth=2)
        plt.legend()
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.title('FIR Filtered Signal')
        plt.show()

    def save_coefficients(self, coefficients, filename='filter_coefficients.txt'):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            np.savetxt(file_path, coefficients, delimiter=',')
            tk.messagebox.showinfo("Saved", "Filter coefficients saved to {}".format(file_path))


def main():
    root = tk.Tk()
    app = FIRFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
