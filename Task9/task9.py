import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import freqz

class FIRFilterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("FIR Filter Application")

        # Initialize variables
        self.filter_type_var = tk.StringVar(value='Low pass')
        self.fs_var = tk.DoubleVar(value=8)
        self.stop_band_attenuation_var = tk.DoubleVar(value=50)
        self.fc_var = tk.DoubleVar(value=1.5)
        self.transition_band_var = tk.DoubleVar(value=.500)
        self.lp_coefficients = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Filter Type:").grid(row=0, column=0, padx=10, pady=5)
        filter_types = ['lowpass', 'highpass', 'bandpass', 'bandstop']
        self.filter_type_menu = tk.OptionMenu(self.master, self.filter_type_var, *filter_types)
        self.filter_type_menu.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(self.master, text="FS:").grid(row=1, column=0, padx=10, pady=5)
        self.fs_entry = tk.Entry(self.master, textvariable=self.fs_var, width=50)
        self.fs_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.master, text="Stop Band Attenuation:").grid(row=2, column=0, padx=10, pady=5)
        self.stop_band_attenuation_entry = tk.Entry(self.master, textvariable=self.stop_band_attenuation_var, width=50)
        self.stop_band_attenuation_entry.grid(row=2, column=1, padx=10, pady=5)

        tk.Label(self.master, text="FC:").grid(row=3, column=0, padx=10, pady=5)
        self.fc_entry = tk.Entry(self.master, textvariable=self.fc_var, width=50)
        self.fc_entry.grid(row=3, column=1, padx=10, pady=5)

        tk.Label(self.master, text="TW:").grid(row=4, column=0, padx=10, pady=5)
        self.transition_band_entry = tk.Entry(self.master, textvariable=self.transition_band_var, width=50)
        self.transition_band_entry.grid(row=4, column=1, padx=10, pady=5)

        # Button to run the FIR filter
        tk.Button(self.master, text="Run FIR Filter", command=self.run_fir_filter).grid(row=5, column=0, columnspan=2, pady=10)
    def calculate_LowPass_HD(self,FCnorm, n):
        if (n == 0):
            result = 2 * FCnorm
        else:    
            result = 2 * FCnorm * ((np.sin(n * 2 * np.pi * FCnorm)) / (n * 2 * np.pi * FCnorm))
        return result

    def calculate_HammingW(self,n, N):
        result = 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)
        return result

    def FIR(self ,N, FCnorm):
        H = []
        indices = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            H.append(self.calculate_LowPass_HD(FCnorm, n) * self.calculate_HammingW(n, N))
            indices.append(n)
        return H, indices

    def run_fir_filter(self):
        filter_type = self.filter_type_var.get()
        fs = float(self.fs_var.get())
        stop_band_attenuation = float(self.stop_band_attenuation_var.get())
        fc = float(self.fc_var.get())
        transition_band = float(self.transition_band_var.get())

        deltaF = transition_band / fs
        N = 3.3 / deltaF
        if int(N) % 2 != 0:
            N = int(N)
        else:
            N = int(np.ceil(3.3 / deltaF))
        FCnormalized = (fc / fs) + (deltaF / 2)
        result, resultIndices = self.FIR(N, FCnormalized)
        self.plot_results(resultIndices, result)
        self.save_coefficients(result)

    def compute_num_taps(self, delta_f, stop_attenuation, fs):
        delta_omega = 2 * np.pi * delta_f / fs
        num_taps = int(6.6 * fs / delta_omega)
        if num_taps % 2 == 0:  # Ensure it is odd
            num_taps += 1
        return num_taps
    def plot_results(self, indices, res):
        plt.plot(indices, res)
        plt.title('FIR Lowpass Filter Frequency Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.grid(True)
        plt.show()

    def save_coefficients(self, coefficients):
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
