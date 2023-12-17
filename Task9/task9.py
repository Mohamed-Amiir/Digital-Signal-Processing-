import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import CompareSignal as test

class FIRFilterApp:

    def __init__(self, master):
        self.master = master
        self.master.title("FIR Filter Application")
        # Initialize variables
        self.filter_type_var = tk.StringVar(value='Low pass')
        self.fs_var = tk.DoubleVar(value=0)
        self.stop_band_attenuation_var = tk.IntVar(value=0)
        self.fc_var = tk.DoubleVar(value=0)
        self.f1_var = tk.DoubleVar(value=0)
        self.f2_var = tk.DoubleVar(value=0)
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

        tk.Label(self.master, text="F1:").grid(row=4, column=0, padx=10, pady=5)
        self.fc_entry = tk.Entry(self.master, textvariable=self.f1_var, width=50)
        self.fc_entry.grid(row=4, column=1, padx=10, pady=5)

        tk.Label(self.master, text="F2:").grid(row=5, column=0, padx=10, pady=5)
        self.fc_entry = tk.Entry(self.master, textvariable=self.f2_var, width=50)
        self.fc_entry.grid(row=5, column=1, padx=10, pady=5)

        tk.Label(self.master, text="TW:").grid(row=6, column=0, padx=10, pady=5)
        self.transition_band_entry = tk.Entry(self.master, textvariable=self.transition_band_var, width=50)
        self.transition_band_entry.grid(row=6, column=1, padx=10, pady=5)
        # Button to load values from file
        load_button = tk.Button(self.master, text="Load from File", command=self.load_values_from_file)
        load_button.grid(row=7, column=0, columnspan=2, pady=10)

        self.coefficients = []
        self.coefficientsIndecies = []
        # Button to run the FIR filter
        tk.Button(self.master, text="Run FIR Filter", command=self.run_fir_filter).grid(row=8, column=0, columnspan=2, pady=10)
        ecg_button = tk.Button(self.master, text="ECG", command=self.ecg)
        ecg_button.grid(row=9, column=0, columnspan=2, pady=10)

    def ecg(self):
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'r') as file:
                file_content = file.read()
        input_data = file_content.split('\n')[3:]
        input_data = [line.split() for line in input_data if line.strip()]
        indices, ecg400 = zip(*[(int(index), float(value)) for index, value in input_data])
        # ind = np.array(indices)
        # ecg400 = np.array(values)

        ecgResult = self.convolve(ecg400, self.coefficients)
        INDCIS = []
        start_index = self.coefficientsIndecies[0]
        end_index = 400 + abs(start_index)
        for i in range(start_index,end_index):
            INDCIS.append(i)

        # indic = np.arange(start_index, end_index + 1)

        # Extract the portion of the convolution result within the specified range
        # ecgResult_subset = ecgResult[start_index:end_index + 1]

        # indices = np.arange(len(ecgResult))

        plt.plot(ecgResult)
        plt.title('ECG')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain (dB)')
        plt.grid(True)
        plt.show()
        messagebox.showinfo("NOW","Save the Result")
        self.save_coefficients(INDCIS,ecgResult)
        messagebox.showinfo("NOW","Check your solution, upload the optimal solution file")
        filePath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        test.Compare_Signals(filePath,INDCIS,ecgResult)

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
                    elif key == 'f1':
                        self.f1_var.set(float(value))
                    elif key == 'f2':
                        self.f2_var.set(float(value))
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
                a = 2 * F2 * ((np.sin(n * 2 * np.pi * F2)) / (n * 2 * np.pi * F2))
                b = -2 * F1 * ((np.sin(n * 2 * np.pi * F1)) / (n * 2 * np.pi * F1))
                x = float(a + b)
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

    def calculte_Blackman(self,N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):  
            x = .42+.5 * np.cos((2 * np.pi * n)/(N-1)) + .08 * np.cos((4 * np.pi * n)/(N - 1))
            result.append(x)    
        return result 

    def FIR(self,filter,window,N, FCnorm,F1norm,F2Norm):
        F = []
        W = []
        output_file = ""
        if(filter == "Low pass"):
            F = self.calculate_LowPass_HD(FCnorm,N)
            output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 1\\LPFCoefficients.txt"
        elif(filter == "High pass"):
            F = self.calculate_HighPass_HD(FCnorm,N) 
            output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 3\\HPFCoefficients.txt"
        elif(filter == "Band pass"):
            F = self.calculate_BandPass_HD(F1norm,F2Norm,N) 
            output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 5\\BPFCoefficients.txt"
        elif(filter == "Band stop"):
            F = self.calculate_BandStop_HD(F1norm,F2Norm,N) 
            output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 7\\BSFCoefficients.txt"

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
            W = self.calculte_Blackman(N)   

        H = []
        indices = []
        for i in range(N):
            H.append(W[i]*F[i])

        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            indices.append(n)
        return H, indices,output_file

    def run_fir_filter(self):
        filter_type = self.filter_type_var.get()
        fs = (self.fs_var.get())/1000
        f1 = (self.f1_var.get())/1000
        f2 = (self.f2_var.get())/1000
        stop_band_attenuation = int(self.stop_band_attenuation_var.get())
        fc = float(self.fc_var.get())/1000
        transition_band = float(self.transition_band_var.get())/1000
        N = 0
        FCnormalized = 0 
        F1Norm = 0
        F2Norm = 0
        window = ""    
        deltaF = 0
        if (filter_type == "Low pass"):
            deltaF = transition_band / fs
            FCnormalized = (fc / fs) + (deltaF / 2)
        elif (filter_type == "High pass"):
            deltaF = transition_band / fs
            FCnormalized = (fc / fs) - (deltaF / 2)
        elif (filter_type == "Band pass"):
            deltaF = transition_band / fs
            F1Norm = (f1 / fs) - (deltaF / 2)
            F2Norm = (f2 / fs) + (deltaF / 2)
        elif (filter_type == "Band stop"):
            deltaF = transition_band / fs
            F1Norm = (f1 / fs) + (deltaF / 2)
            F2Norm = (f2 / fs) - (deltaF / 2)

        if(stop_band_attenuation < 21):
            window = "rectangular"
            N = 0.9 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            else:
                N = int(np.ceil(0.9 / deltaF))
        elif(stop_band_attenuation < 44):
            window = "hanning"
            N = 3.1 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            else:
                N = int(np.ceil(3.1 / deltaF))
        elif( stop_band_attenuation < 53 ):
            window = "hamming"  
            N = 3.3 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            else:
                N = int(np.ceil(3.3 / deltaF))
        elif(stop_band_attenuation < 74):
            window = "blackman"  
            N = 5.5 / deltaF
            if int(N) % 2 != 0:
                N = int(N)
            elif N % 2 == 0:
                N = N + 1 
                N = int(N)     
            else:
                N = int(np.ceil(5.5 / deltaF))    
        result, resultIndices,outputFile = self.FIR(filter_type,window,N, FCnormalized,F1Norm,F2Norm)
        self.coefficients = result
        self.coefficientsIndecies = resultIndices
        # messagebox.showinfo("NOW","Check your solution, upload the optimal solution file")
        test.Compare_Signals(outputFile,resultIndices,result)
        self.plot_results(resultIndices, result)
        messagebox.showinfo("NOW","Save your solution")
        self.save_coefficients(resultIndices,result)
    
    def convolve(self,a,b):
        lenA,lenB = len(a) ,len(b)
        result = [0]*(lenA+lenB-1)
        for i in range(lenA):
            for j in range(lenB):
                result[i+j] += a[i]*b[j]
        return result
    
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