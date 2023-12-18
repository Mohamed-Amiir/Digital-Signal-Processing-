import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import dct

class ECGProcessor:
    def __init__(self, root):
        self.root = root
        self.root.geometry("500x500")
        self.root.title("ECG Signal Processor")

        self.subject_a_path = ""
        self.subject_b_path = ""
        self.test_path = ""

        self.fs_label = tk.Label(root, text="Sampling Frequency (Fs):")
        self.fs_label.pack()
        self.fs_entry = tk.Entry(root)
        self.fs_entry.pack()

        self.miniF_label = tk.Label(root, text="Minimum Frequency (miniF):")
        self.miniF_label.pack()
        self.miniF_entry = tk.Entry(root)
        self.miniF_entry.pack()

        self.maxF_label = tk.Label(root, text="Maximum Frequency (maxF):")
        self.maxF_label.pack()
        self.maxF_entry = tk.Entry(root)
        self.maxF_entry.pack()

        self.newFs_label = tk.Label(root, text="New Sampling Frequency (newFs):")
        self.newFs_label.pack()
        self.newFs_entry = tk.Entry(root)
        self.newFs_entry.pack()

        self.load_subject_a_button = tk.Button(root, text="Load Subject A", command=self.load_subject_a)
        self.load_subject_a_button.pack()

        self.load_subject_b_button = tk.Button(root, text="Load Subject B", command=self.load_subject_b)
        self.load_subject_b_button.pack()

        self.load_test_button = tk.Button(root, text="Load Test", command=self.load_test)
        self.load_test_button.pack()

        self.process_button = tk.Button(root, text="Process Data", command=self.process_data)
        self.process_button.pack()

        # Buttons for each process
        self.filter_button = tk.Button(root, text="1. Filter Signal", command=self.filter_signal)
        self.filter_button.pack()

        self.resample_button = tk.Button(root, text="2. Resample Signal", command=self.resample_signal)
        self.resample_button.pack()

        self.remove_dc_button = tk.Button(root, text="3. Remove DC Component", command=self.remove_dc_component)
        self.remove_dc_button.pack()

        self.normalize_button = tk.Button(root, text="4. Normalize Signal", command=self.normalize_signal)
        self.normalize_button.pack()

        self.auto_corr_button = tk.Button(root, text="5. Compute Auto Correlation", command=self.compute_auto_correlation)
        self.auto_corr_button.pack()

        self.preserve_coeff_button = tk.Button(root, text="6. Preserve Coefficients", command=self.preserve_coefficients)
        self.preserve_coeff_button.pack()

        self.dct_button = tk.Button(root, text="7. Compute DCT", command=self.compute_dct)
        self.dct_button.pack()

        self.template_match_button = tk.Button(root, text="8. Template Matching", command=self.template_matching)
        self.template_match_button.pack()

        self.display_results_button = tk.Button(root, text="9. Display Results", command=self.display_results)
        self.display_results_button.pack()
        a1 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg1.txt"
        a2 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg2.txt"
        a3 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg3.txt"
        a4 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg4.txt"
        a5 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg5.txt"
        a6 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg6.txt"
        
        b1 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg1.txt"
        b2 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg2.txt"
        b3 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg3.txt"
        b4 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg4.txt"
        b5 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg5.txt"
        b6 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg6.txt"
        self.Apath = [a1,a2,a3,a4,a5,a6]
        self.Bpath = [b1,b2,b3,b4,b5,b6]
        self.A = []
        self.B = []
        for path in self.A:
            lines = path.readlines()
            data_array = [float(line.strip()) for line in lines]
            self.A.append(data_array)
        for path in self.B:
            lines = path.readlines()
            data_array = [float(line.strip()) for line in lines]
            self.B.append(data_array)




    def Run(self):
        ####### FILTERING #######
        for seg in self.A:
            seg = self.filter_signal(seg)
        for seg in self.B:
            seg = self.filter_signal(seg)
        #########################
        ####### Resampling ######
        for seg in self.A:
            seg = self.resampling(seg)
        for seg in self.B:
            seg = self.resampling(seg)
        #########################
        ####### Remove DC #######
        for seg in self.A:
            seg = self.remove_dc(seg)
        for seg in self.B:
            seg = self.remove_dc(seg)
        #########################
        ####### Normalize #######
        for seg in self.A:
            seg = self.normalize_signal(seg)
        for seg in self.B:
            seg = self.normalize_signal(seg)
        #########################
        ###### Correlation ######
        for seg in self.A:
            seg = self.calculate_correlation(seg)
        for seg in self.B:
            seg = self.calculate_correlation(seg)
        #########################
            






    def calculate_correlation(self,data):
        signal1 = data
        signal2 = data

        corelation = []
        normCorealtion = []
        for n in range(len(signal1) + 1):
            if n == 0:
                continue
            else:
                signal2.append(signal2[0])
                signal2.remove(signal2[0])
            r = 0
            p = 0
            for i in range(len(signal1)):
                r += (1 / len(signal1)) * (signal1[i] * signal2[i])
            corelation.append(r)
            sig1 = 0
            sig2 = 0
            for j in range(len(signal1)):
                sig1 += signal1[j] * signal1[j]
                sig2 += signal2[j] * signal2[j]
            p = r / ((1 / len(signal1)) * np.power((sig1 * sig2), .5))
            normCorealtion.append(p)
        x = []
        x.append(corelation[len(corelation)-1])
        for i in range(len(corelation)-1):
            x.append(corelation[i])  
        y = []
        y.append(normCorealtion[len(normCorealtion)-1])
        for i in range(len(normCorealtion)-1):
            y.append(normCorealtion[i])      

        corelation = x
        normCorealtion = y

        # plt.subplot(1, 3, 1)
        # plt.plot(signal1, marker='o')
        # plt.plot(signal2, marker='o')
        # plt.title("Original")

        # plt.subplot(1, 3, 2)
        # plt.plot(corelation,marker='o')
        # plt.title("Cross Correlation")

        # plt.subplot(1,3,3)
        # plt.plot(normCorealtion)
        # plt.title("Normalized Cross Correlation")

        # plt.tight_layout()
        # plt.show()
        return normCorealtion

              
    def load_subject_a(self):
        self.subject_a_path = filedialog.askdirectory()
        print("Subject A loaded:", self.subject_a_path)
    def load_subject_b(self):
        self.subject_b_path = filedialog.askdirectory()
        print("Subject B loaded:", self.subject_b_path)
    def load_test(self):
        self.test_path = filedialog.askdirectory()
        print("Test data loaded:", self.test_path)
    def process_data(self):
        Fs = float(self.fs_entry.get())
        miniF = float(self.miniF_entry.get())
        maxF = float(self.maxF_entry.get())
        newFs = float(self.newFs_entry.get())

        # Add your general processing logic here
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
    def calculate_Hamming(self, N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):  
            x = 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)
            result.append(x)    
        return result   
    def upsample(self,signal, factor):
        result = []
        for element in signal:
            result.extend([element] + [0] * (factor-1))
        for i in range(factor-1):
            result.pop()    
        return result
    def downsample(self,signal, factor):
        return signal[::factor]
    def run_fir_filter(self):
        filter_type = "Low pass"
        fs = 8000/1000
        f1 =  0/1000
        f2 = 0/1000
        stop_band_attenuation = int(50)
        fc = float(1500)/1000
        transition_band = float(500)/1000
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
        
        result, resultIndices, outputFile = self.FIR(filter_type,window,N, FCnormalized,F1Norm,F2Norm)
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

    def resampling(self,input_signal):
        filtered_signal = self.run_fir_filter()
        Fs = float(self.fs_entry.get())      
        newFs = float(self.newFs_entry.get())
        M = newFs/Fs
        L = Fs / newFs
        ###########################################################
        if M == 0 and L == 0:
            return "Error: Both M and L cannot be zero."
        if M == 0:
            x = self.upsample(input_signal, L)
            resampled_signal = self.convolve(x,filtered_signal)
        elif L == 0:
            x = self.convolve(filtered_signal,input_signal)
            resampled_signal = self.downsample(x, M)
        else:
            # Fractional rate change: upsample, filter, downsample
            upsampled_signal = self.upsample(input_signal, L)
            x = self.convolve(filtered_signal,upsampled_signal)
            resampled_signal = self.downsample(x, M)
        indecis =  []
        for i in range(int(-(len(filtered_signal)/2)),int( len(resampled_signal) - (len(filtered_signal)/2))+1):
            indecis.append(i)
        
        # print(resampled_signal)
        # print(len(resampled_signal))
        # Display the result
        # plt.plot(input_signal, label="Original Signal")
        # plt.plot(resampled_signal, label="Resampled Signal")
        # plt.title("Original vs. Resampled Signal")
        # plt.xlabel("Sample Index")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.show()
        return resampled_signal
    def filter_signal(self,seg): #DONE
        print("Filtering signal...")
        stopbandattenuation = 50
        transition_band = 500/1000
        fs = float(self.fs_entry.get())  
        miniF = float(self.miniF_entry.get())
        maxF = float(self.maxF_entry.get())
        newFs = float(self.newFs_entry.get())  
        deltaF = transition_band / fs
        F1Norm = (miniF / fs) - (deltaF / 2)
        F2Norm = (maxF / fs) + (deltaF / 2)
        N = 3.3 / deltaF
        if int(N) % 2 != 0:
            N = int(N)
        elif N % 2 == 0 :
            N =N +1
            N = int(N)  
        else:
            N = int(np.ceil(3.3 / deltaF))

        W = []
        F = []
        F = self.calculate_BandPass_HD(F1Norm,F2Norm,N)
        W = self.calculate_Hamming(N)
        
        H = []
        indices = []
        for i in range(N):
            H.append(W[i]*F[i])
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            indices.append(n)
        return self.convolve(H,seg)
    
    def resample_signal(self):
        print("Resampling signal...")

    def remove_dc(self,Data):
        # file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\Lab 5\\Task files\Remove DC component\\DC_component_input.txt"
        # with open(file_name, 'r') as f:
        #     data = [line.split() for line in f.read().split('\n') if line.strip()]

        # values = [float(value) for _, value in data[3:]]  # Skip the first 3 lines
        # Data = np.array(values)
        sum = 0
        for element in Data:
            sum += element
        average = sum / len(Data)
        result= []
        for element in Data:
            result.append(round((element-average),3))
        # print("Original: ",Data)
        # print("Result: ",result)# CORRECT BUT you need to take just first 3 decimal numbers to get ACCEPTED
        # plt.subplot(1, 2, 1)
        # plt.plot(Data, marker='o')
        # plt.xlabel("Sample Index")
        # plt.ylabel("Amplitude")
        # plt.title("Original")

        # plt.subplot(1, 2, 2)
        # plt.plot(result,marker='o')
        # plt.xlabel("Sample Index")
        # plt.ylabel("Amplitude")
        # plt.title("After Removing")

        # plt.tight_layout()
        # plt.show()
        return result
    def remove_dc_component(self):
        print("Removing DC component...")

    def normalize_signal(self,data):
        normalized_signal = [(sample[0], (sample[1] - min_amplitude) / (max_amplitude - min_amplitude)) for sample in data]
        max_amplitude = max(sample[1] for sample in data)
        min_amplitude = min(sample[1] for sample in data)
        if max_amplitude == min_amplitude:
            return data  # Avoid division by zero
        normalized_signal = [(sample[0], 2 * (sample[1] - min_amplitude) / (max_amplitude - min_amplitude) - 1) for sample in data]
        return normalized_signal

    def compute_auto_correlation(self):
        # Add your compute auto correlation logic here
        print("Computing Auto Correlation...")

    def preserve_coefficients(self):
        # Add your preserve coefficients logic here
        print("Preserving Coefficients...")

    def compute_dct(self):
        # Add your compute DCT logic here
        print("Computing DCT...")

    def template_matching(self):
        # Add your template matching logic here
        print("Template Matching...")

    def display_results(self):
        # Add your display results logic here
        print("Displaying Results...")
    def convolve(self,a,b):
            lenA,lenB = len(a) ,len(b)
            result = [0]*(lenA+lenB-1)
            for i in range(lenA):
                for j in range(lenB):
                    result[i+j] += a[i]*b[j]
            return result

if __name__ == "__main__":
    root = tk.Tk()
    ecg_processor = ECGProcessor(root)
    root.mainloop()
