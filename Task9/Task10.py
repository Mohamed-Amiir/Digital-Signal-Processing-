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
        self.fs_var = tk.DoubleVar(value=8)
        self.MiniF = tk.DoubleVar(value=2)
        self.MaxF = tk.DoubleVar(value=10)
        self.NewF = tk.DoubleVar(value=12)
        
        self.subject_a_path = ""
        self.subject_b_path = ""
        self.test_path = ""
        self.A = []
        self.B = []
        
        self.create_widgets()
        self.process_data()


    def create_widgets(self):
        self.fs_label = tk.Label(root, text="Sampling Frequency (Fs):")
        self.fs_label.pack()
        self.fs_entry = tk.Entry(self.root, textvariable=self.fs_var, width=20)
        self.fs_entry.pack()

        self.miniF_label = tk.Label(root, text="Minimum Frequency (miniF):")
        self.miniF_label.pack()
        self.miniF_entry = tk.Entry(self.root, textvariable=self.MiniF, width=20)
        self.miniF_entry.pack()

        self.maxF_label = tk.Label(root, text="Maximum Frequency (maxF):")
        self.maxF_label.pack()
        self.maxF_entry = tk.Entry(self.root, textvariable=self.MaxF, width=20)
        self.maxF_entry.pack()

        self.newFs_label = tk.Label(root, text="New Sampling Frequency (newFs):")
        self.newFs_label.pack()
        self.newFs_entry = tk.Entry(self.root, textvariable=self.NewF, width=20)
        self.newFs_entry.pack()

        self.load_subject_a_button = tk.Button(root, text="Run", command=self.Run)
        self.load_subject_a_button.pack()

        # self.load_subject_b_button = tk.Button(root, text="Load Subject B", command=self.load_subject_b)
        # self.load_subject_b_button.pack()

        # self.load_test_button = tk.Button(root, text="Load Test", command=self.load_test)
        # self.load_test_button.pack()

        # self.process_button = tk.Button(root, text="Process Data", command=self.process_data)
        # self.process_button.pack()

        # Buttons for each process
        # self.filter_button = tk.Button(root, text="1. Filter Signal", command=self.filter_signal)
        # self.filter_button.pack()

        # self.resample_button = tk.Button(root, text="2. Resample Signal", command=self.resample_signal)
        # self.resample_button.pack()

        # self.remove_dc_button = tk.Button(root, text="3. Remove DC Component", command=self.remove_dc_component)
        # self.remove_dc_button.pack()

        # self.normalize_button = tk.Button(root, text="4. Normalize Signal", command=self.normalize_signal)
        # self.normalize_button.pack()

        # self.auto_corr_button = tk.Button(root, text="5. Compute Auto Correlation", command=self.compute_auto_correlation)
        # self.auto_corr_button.pack()

        # self.preserve_coeff_button = tk.Button(root, text="6. Preserve Coefficients", command=self.preserve_coefficients)
        # self.preserve_coeff_button.pack()

        # self.dct_button = tk.Button(root, text="7. Compute DCT", command=self.compute_dct)
        # self.dct_button.pack()

        # self.template_match_button = tk.Button(root, text="8. Template Matching", command=self.template_matching)
        # self.template_match_button.pack()

        # self.display_results_button = tk.Button(root, text="9. Display Results", command=self.display_results)
        # self.display_results_button.pack()
    def template_matching(self):
        down = []
        up = []
        class11 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 1\\down1.txt"
        class12 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 1\\down2.txt"
        class13 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 1\\down3.txt"
        class14 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 1\\down4.txt"
        class15 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 1\\down5.txt"

        class21 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 2\\up1.txt"
        class22 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 2\\up2.txt"
        class23 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 2\\up3.txt"
        class24 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 2\\up4.txt"
        class25 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab7\\SC and Csys\\Task Files\\point3 Files\\Class 2\\up5.txt"
        class1 = [class11, class12, class13, class14, class15]
        class2 = [class21, class22, class23, class24, class25]


        for i in range(5):
            with open(class1[i], 'r') as file:
                content = file.read()
            values_str = content.split('\n')
            values_int = [int(value) for value in values_str if value]
            values_array = np.array(values_int)
            down.append(values_array)
        for i in range(5):
            with open(class2[i], 'r') as file:
                content = file.read()
            values_str = content.split('\n')
            values_int = [int(value) for value in values_str if value]
            values_array = np.array(values_int)
            up.append(values_array)


        downAvg = []
        upAvg = []
       
        for i in range(251):
            x = 0
            y = 0
            for j in range(5):
                x += down[j][i]
                y += up[j][i]
            a = x / 5
            a2 = y / 5
            downAvg.append(a)
            upAvg.append(a2)
        # NOW WE HAVE THE AVG OF CLASS 1 AND THE AVG OF CLASS 2
        # NOW WE WILL COLLERATE THE TEST WITH BOTH OF THEM AND DETECT THE HIGHEST CORRELATION
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        with open(file_path, 'r') as file:
            content = file.read()
        values_str = content.split('\n')
        test = [int(value) for value in values_str if value]
        # test = np.array(values_int)
        test2 = test
        # NOW WE WILL COLERRATE ( downAvg and test ) and ( upAvg and test )
        # THEN WE GET TWO ARRAYS OF COLERRATION , THEN WE WILL MAXIMIZE BETWEEN THE MAXIMUM VALUES OF EACH ARRAY
        # print("Hello world")
        class1Corelation = []
        class1NormCorealtion = []
        for n in range(len(downAvg) + 1):
            if n == 0:
                continue
            else:
                test.append(test[0])
                test.remove(test[0])
            r = 0
            p = 0
            for i in range(len(downAvg)):
                r += (1 / len(downAvg)) * (downAvg[i] * downAvg[i])
            class1Corelation.append(r)
            sig1 = 0
            sig2 = 0
            for j in range(len(downAvg)):
                sig1 += downAvg[j] * downAvg[j]
                sig2 += test[j] * test[j]
            p = r / ((1 / len(downAvg)) * np.power((sig1 * sig2), .5))
            class1NormCorealtion.append(p)


        class2Corelation = []
        class2NormCorealtion = []
        for n in range(len(upAvg) + 1):
            if n == 0:
                continue
            else:
                test2.append(test2[0])
                test2.remove(test2[0])
            r = 0
            p = 0
            for i in range(len(upAvg)):
                r += (1 / len(upAvg)) * (upAvg[i] * upAvg[i])
            class2Corelation.append(r)
            sig1 = 0
            sig2 = 0
            for j in range(len(upAvg)):
                sig1 += downAvg[j] * downAvg[j]
                sig2 += test[j] * test[j]
            p = r / ((1 / len(upAvg)) * np.power((sig1 * sig2), .5))
            class2NormCorealtion.append(p)

        CLASS1 = np.argmax(class1NormCorealtion)
        CLASS2 = np.argmax(class2NormCorealtion)
        if CLASS1 > CLASS2:
            print("Class 1")
        else:
            print("Class 2")


    def Run(self):
        FILEA ="C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\Practical task 2\\Test Folder\\ATest1.txt"
        with open(FILEA, 'r') as file:
            lines = file.readlines()
        Atest = [float(line.strip()) for line in lines]

        Atest = self.filter_signal(Atest)
        Atest = self.resampling(Atest)
        Atest = self.remove_dc(Atest)
        Atest = self.normalize_signal(Atest)
        Atest = self.calculate_correlation(Atest)
        FILEB ="C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\Practical task 2\\Test Folder\\BTest1.txt"
        with open(FILEA, 'r') as file:
            lines = file.readlines()
        Atest = [float(line.strip()) for line in lines]

        Btest = self.filter_signal(Atest)
        Btest = self.resampling(Atest)
        Btest = self.remove_dc(Atest)
        Btest = self.normalize_signal(Atest)
        Btest = self.calculate_correlation(Atest)

        print("Filtering.......")
        ####### FILTERING #######
        for i in range(6):
            self.A[i] = self.filter_signal(self.A[i])
        for i in range(6):
            self.B[i] = self.filter_signal(self.B[i])
        #########################
        print("Resampling......")
        ####### Resampling ######
        for i in range(6):
            self.A[i] = self.resampling(self.A[i])
        for i in range(6):
            self.B[i] = self.resampling(self.B[i])
        #########################
        print("Removing DC.....")
        ####### Remove DC #######
        for i in range(6):
            self.A[i] = self.remove_dc(self.A[i])
        for i in range(6):
            self.B[i] = self.remove_dc(self.B[i])
        #########################
        print("Normalizing.....")    
        ####### Normalize #######
        for i in range(6):
            self.A[i] = self.normalize_signal(self.A[i])
        for i in range(6):
            self.B[i] = self.normalize_signal(self.B[i])
        #########################
        print("Correlation.....")    
        ###### Correlation ######
        for seg in self.A:
            seg = self.calculate_correlation(seg)
        for seg in self.B:
            seg = self.calculate_correlation(seg)
        #########################
        print("We are done")





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
        Apath = [a1, a2, a3, a4, a5, a6]
        Bpath = [b1, b2, b3, b4, b5, b6]
        

        for path in Apath:
            with open(path, 'r') as file:
                lines = file.readlines()
            data_array = [float(line.strip()) for line in lines]
            self.A.append(data_array)

        for path in Bpath:
            with open(path, 'r') as file:
                lines = file.readlines()
            data_array = [float(line.strip()) for line in lines]
            self.B.append(data_array)


        # Add your general processing logic here
    def calculate_BandPass_HD(self,F1,F2, N):
        result = []
        for n in range(int(-(N // 2)), int((N // 2) + 1)):
            x = 0
            if (n == 0):
                x = 2 * (F2-F1)
            else: 
                a = 2 * F2 * ((np.sin(n * 2 * np.pi * F2)) / (n * 2 * np.pi * F2)) if (n * 2 * np.pi * F2) != 0 else 0
                b = -2 * F1 * ((np.sin(n * 2 * np.pi * F1)) / (n * 2 * np.pi * F1)) if (n * 2 * np.pi * F1) != 0 else 0

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
            F = self.calculate_BandPass_HD(F1norm,F2Norm,N) 
            W = self.calculate_Hamming(N)   
            output_file = ""
            # if(filter == "Low pass"):
            #     F = self.calculate_LowPass_HD(FCnorm,N)
            #     output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 1\\LPFCoefficients.txt"
            # elif(filter == "High pass"):
            #     F = self.calculate_HighPass_HD(FCnorm,N) 
            #     output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 3\\HPFCoefficients.txt"
            # elif(filter == "Band pass"):
            #     F = self.calculate_BandPass_HD(F1norm,F2Norm,N) 
            #     output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 5\\BPFCoefficients.txt"
            # elif(filter == "Band stop"):
            #     F = self.calculate_BandStop_HD(F1norm,F2Norm,N) 
            #     output_file = "C:\\Users\\lenovo\\Desktop\\My-Github\\DSP\\Task9\\FIR test cases\\Testcase 7\\BSFCoefficients.txt"

            # if(window == "rectangular"):
            #     W = self.calculate_Hamming(N)
            # elif(window == "hanning"):
            #     W = self.calculte_Haning(N)   
            # elif(window == "hamming"):
            #     W = self.calculate_Hamming(N)   
            # elif(window == "blackman"):
            #     W = self.calculte_Blackman(N)   

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
        M = int(newFs/Fs)
        L = int(Fs / newFs)
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
        x_list = np.array([])
        mean_value = np.mean(Data)
        for x_val in Data:
            x_list = np.append(x_list, x_val - mean_value)
        return x_list

    # def remove_dc(self,Data):
    #     # file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\Lab 5\\Task files\Remove DC component\\DC_component_input.txt"
    #     # with open(file_name, 'r') as f:
    #     #     data = [line.split() for line in f.read().split('\n') if line.strip()]

    #     # values = [float(value) for _, value in data[3:]]  # Skip the first 3 lines
    #     # Data = np.array(values)
    #     sum = 0
    #     for element in Data:
    #         sum += element
    #     average = sum / len(Data)
    #     result= []
    #     for element in Data:
    #         result.append(round((element-average),3))
    #     # print("Original: ",Data)
    #     # print("Result: ",result)# CORRECT BUT you need to take just first 3 decimal numbers to get ACCEPTED
    #     # plt.subplot(1, 2, 1)
    #     # plt.plot(Data, marker='o')
    #     # plt.xlabel("Sample Index")
    #     # plt.ylabel("Amplitude")
    #     # plt.title("Original")

    #     # plt.subplot(1, 2, 2)
    #     # plt.plot(result,marker='o')
    #     # plt.xlabel("Sample Index")
    #     # plt.ylabel("Amplitude")
    #     # plt.title("After Removing")

    #     # plt.tight_layout()
    #     # plt.show()
    #     return result
    # def remove_dc_component(self):
        print("Removing DC component...")

    def normalize_signal(self,signal):
        max_abs_value = np.max(np.abs(signal))

        if max_abs_value != 0:
            normalized_signal = signal / max_abs_value
        else:
            normalized_signal = signal

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
