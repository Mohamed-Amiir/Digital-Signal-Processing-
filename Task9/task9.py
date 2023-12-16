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
        self.fs_var = tk.DoubleVar(value=0)
        self.stop_band_attenuation_var = tk.IntVar(value=0)
        self.fc_var = tk.DoubleVar(value=0)
        self.transition_band_var = tk.DoubleVar(value=0)
        self.lp_coefficients = None

        # Create GUI elements
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.master, text="Filter Type:").grid(row=0, column=0, padx=10, pady=5)
        filter_types = ['Low pass', 'High pass', 'Band pass', 'Band stop']
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
          # Button to load values from file
        load_button = tk.Button(self.master, text="Load from File", command=self.load_values_from_file)
        load_button.grid(row=5, column=0, columnspan=2, pady=10)

        # Button to run the FIR filter
        tk.Button(self.master, text="Run FIR Filter", command=self.run_fir_filter).grid(row=6, column=0, columnspan=2, pady=10)
    def load_values_from_file(self):
        file_path = filedialog.askopenfilename(title="Select Input File", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                for line in file:
                    key, value = line.strip().split('=')
                    key = key.strip().lower()
                    value = value.strip()
                    if key == 'filtertype':
                        self.filter_type_var.set(value)
                    elif key == 'fs':
                        self.fs_var.set(float(value))
                    elif key == 'stopbandattenuation':
                        self.stop_band_attenuation_var.set(float(value))
                    elif key == 'fc':
                        self.fc_var.set(float(value))
                    elif key == 'transitionband':
                        self.transition_band_var.set(float(value))    
    
    def calculate_LowPass_HD(self,FCnorm, N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            x = 0
            if (n == 0):
                x = 2 * FCnorm
            else:    
                x = 2 * FCnorm * ((np.sin(n * 2 * np.pi * FCnorm)) / (n * 2 * np.pi * FCnorm))
            result.append(x)    
        
        return result
    def calculate_HighPass_HD(self,FCnorm,  N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            x = 0
            if (n == 0):
                x = 1 - (2 * FCnorm)
            else:    
                x = -2 * FCnorm * ((np.sin(n * 2 * np.pi * FCnorm)) / (n * 2 * np.pi * FCnorm))
            result.append(x)    
        
        return result  
    def calculate_BandPass_HD(self,F1,F2, N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            x = 0
            if (n == 0):
                x = 2 * (F2-F1)
            else:    
                x = (2 * F2 * ((np.sin(n * 2 * np.pi * F2)) / (n * 2 * np.pi * F2)))+(-2 * F1 * ((np.sin(n * 2 * np.pi * F1)) / (n * 2 * np.pi * F1)))
            result.append(x)    
        
        return result 
    def calculate_BandStop_HD(self,F1,F2, N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            x = 0
            if (n == 0):
                x = 1 - (2 * (F2-F1))
            else:    
                x = (2 * F1 * ((np.sin(n * 2 * np.pi * F1)) / (n * 2 * np.pi * F1)))+(-2 * F2 * ((np.sin(n * 2 * np.pi * F2)) / (n * 2 * np.pi * F2)))
            result.append(x)    
        
        return result 
    
    def calculate_Hamming(self, N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):  
            x = 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)
            result.append(x)    
        return result 
        
    def calculte_Haning(self,N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):  
            x = 0.5 + 0.5 * np.cos((2 * np.pi * n) / N)
            result.append(x)    
        return result 

    def FIR(self,filter,window,N, FCnorm):
        F = []
        W = []
        if(filter == "Low pass"):
            F = self.calculate_LowPass_HD(FCnorm,N)
        elif(filter == "High pass"):
            F = self.calculate_HighPass_HD(FCnorm,N)   
        # elif(filter == "bandpass"):
        #     F = self.calculate_BandPass_HD(FCnorm,N)   
        # elif(filter == "lowpass"):
        #     F = self.calculate_BandStop_HD(FCnorm,N)

        if(window == "rectangular"):
            W = self.calculate_Hamming(N)
        elif(window == "hanning"):
            W = self.calculte_Haning(N)   
        elif(window == "hamming"):
            W = self.calculate_Hamming(N)   
        elif(window == "blackman"):
            W = self.calculte_Haning(N)   

        H = []
        indices = []
        for i in range(N):
            H.append(W[i]*F[i])

        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            indices.append(n)
        return H, indices

    def run_fir_filter(self):
        filter_type = self.filter_type_var.get()
        fs = float(self.fs_var.get())/1000
        stop_band_attenuation = int(self.stop_band_attenuation_var.get())
        fc = float(self.fc_var.get())/1000
        transition_band = float(self.transition_band_var.get())/1000
        N = 0
        FCnormalized = 0 
        window = ""    
        if(stop_band_attenuation < 13):
            window = "rectangular"
            deltaF = transition_band / fs
            N = 0.9 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            else:
                N = int(np.ceil(3.3 / deltaF))
            FCnormalized = (fc / fs) + (deltaF / 2)
        elif(stop_band_attenuation < 31 & stop_band_attenuation > 13):
            window = "hanning"
            deltaF = transition_band / fs
            N = 3.1 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            else:
                N = int(np.ceil(3.3 / deltaF))
            FCnormalized = (fc / fs) + (deltaF / 2)
        elif(stop_band_attenuation > 41 ):
            window = "hamming"  
            deltaF = transition_band / fs
            N = 3.3 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            else:
                N = int(np.ceil(3.3 / deltaF))
            FCnormalized = (fc / fs) + (deltaF / 2)
    
        result, resultIndices = self.FIR(filter_type,window,N, FCnormalized)
        self.plot_results(resultIndices, result)
        self.save_coefficients(resultIndices,result)
    
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
    def save_coefficients(self, indecis, coefficients):
        if len(indecis) != len(coefficients):
            messagebox.showerror("Error", "Lengths of 'indecis' and 'coefficients' must be the same.")
            return
    
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            # Combine indecis and coefficients vertically
            data_to_save = np.column_stack((indecis, coefficients))
            np.savetxt(file_path, data_to_save, fmt='%d %.10f', delimiter=' ', newline='\n')
            messagebox.showinfo("Saved", "Data saved to {}".format(file_path))


def main():
    root = tk.Tk()
    app = FIRFilterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
