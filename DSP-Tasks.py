import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import comparesignals as cmp
import Shift_Fold_Signal as SFS
import ConvTest as CnvTest
from scipy.fft import fft, ifft
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
import CompareSignal as test

#***************************** TASK 1 ******************************
def browse_file_1():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        # Extract the file name from the path
        selected_file = file_path.split("/")[-1]
        # selected_file = file_path.split("\\")[-1]
        selected_file_label["text"] = selected_file
        data = read_signal_data_1(file_path)
        if data is not None:
            plot_signal_1(data)
def read_signal_data_1(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

        if len(lines) < 3:
            print("Invalid file format.")
            return None

        num_samples = int(lines[2])
        data = []

        for i in range(3, 3 + num_samples):
            parts = lines[i].split()
            if len(parts) != 2:
                print("Invalid line format.")
                return None
            index, amplitude = float(parts[0]), float(parts[1])
            data.append((index, amplitude))

        return data

    except Exception as e:
        print("Error reading the file:", e)
        return None
def plot_signal_1(data):
    if len(data) == 0:
        print("No data to plot.")
        return

    index = [sample[0] for sample in data]
    amplitude = [sample[1] for sample in data]

    plt.figure(figsize=(10, 5))

    # Continuous representation
    plt.subplot(1, 2, 1)
    plt.plot(index, amplitude, label="Continuous Signal", marker='o')
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Continuous Signal Representation")
    plt.legend()

    # Discrete representation
    plt.subplot(1, 2, 2)
    plt.stem(index, amplitude, linefmt='-b', markerfmt='ob', basefmt=' ', label="Discrete Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Discrete Signal Representation")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
def generate_signal_1():

    plt.figure(figsize=(12,5))

    signal_type = signal_type_combobox.get()
    A = float(A_entry.get())
    analog_frequency = float(analog_frequency_entry.get())
    sampling_frequency = float(sampling_frequency_entry.get())
    phase_shift = float(phase_shift_entry.get())

    t = np.arange(0, 1, 1 / sampling_frequency)  # Time vector

    if signal_type == "sin":
        signal = A * np.sin(2 * np.pi * analog_frequency * t + phase_shift)
        signal_label = "Sinusoidal Signal"
    elif signal_type == "cos":
        signal = A * np.cos(2 * np.pi * analog_frequency * t + phase_shift)
        signal_label = "Cosinusoidal Signal"
    else:
        print("Invalid signal type.")
        return

    # Continuous representation
    plt.subplot(1, 2, 1)
    plt.plot(t, signal, label="")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title("Continuous Signal Representation")
    plt.legend()

    # Discrete representation
    plt.subplot(1, 2, 2)
    plt.stem(t, signal, linefmt='-b', markerfmt='ob', basefmt=' ', label="")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Discrete Signal Representation")
    plt.grid(True)

    plt.legend()


    plt.show()
def Task1():
    # main window
    root = tk.Tk()
    root.geometry("800x500")
    root.title("DSP Task 1")

    header_label = ttk.Label(root, text="DSP Task 1", font=("Arial", 20))
    header_label.pack(pady=20)

    upload_button = ttk.Button(root, text="Upload Text File", command=browse_file_1)
    upload_button.pack()

    # display the selected file name
    selected_file_label = ttk.Label(root)
    selected_file_label.pack(pady=10)

    # combo box to select signal type
    signal_type_combobox = ttk.Combobox(root, values=("sin", "cos"))
    signal_type_combobox.pack(pady=10)

    # entry fields for input
    A_label = ttk.Label(root, text="A:")
    A_label.pack()
    A_entry = ttk.Entry(root)
    A_entry.insert(0, "")
    A_entry.pack()

    analog_frequency_label = ttk.Label(root, text="Analog Frequency:")
    analog_frequency_label.pack()
    analog_frequency_entry = ttk.Entry(root)
    analog_frequency_entry.insert(0, "")
    analog_frequency_entry.pack()

    sampling_frequency_label = ttk.Label(root, text="Sampling Frequency:")
    sampling_frequency_label.pack()
    sampling_frequency_entry = ttk.Entry(root)
    sampling_frequency_entry.insert(0, "")
    sampling_frequency_entry.pack()

    phase_shift_label = ttk.Label(root, text="Phase Shift:")
    phase_shift_label.pack()
    phase_shift_entry = ttk.Entry(root)
    phase_shift_entry.insert(0, "")
    phase_shift_entry.pack()

    # Create a button to generate the signal
    generate_button = ttk.Button(root, text="Generate Signal", command=generate_signal_1)
    generate_button.pack(pady=20)

    root.mainloop()
#*******************************************************************
#***************************** TASK 2 ******************************
def browse_file_2(button_number):
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        # Extract the file name from the path
        selected_file = file_path.split("/")[-1]
        selected_files[button_number]["text"] = selected_file  # Display the selected file name
        data = read_signal_data_2(file_path)
        if button_number == 1:
            data1.extend(data)
        else:
            data2.extend(data)
def read_signal_data_2(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

        if len(lines) < 3:
            print("Invalid file format.")
            return None

        signal_type = int(lines[0])
        is_periodic = int(lines[1])
        num_samples = int(lines[2])
        data = []

        for i in range(3, 3 + num_samples):
            parts = lines[i].split()
            if signal_type == 0:  # Time domain
                if len(parts) != 2:
                    print("Invalid line format in time domain signal.")
                    return None
                index, amplitude = float(parts[0]), float(parts[1])
            elif signal_type == 1:  # Frequency domain
                if len(parts) != 3:
                    print("Invalid line format in frequency domain signal.")
                    return None
                index, amplitude, phase_shift = float(parts[0]), float(parts[1]), float(parts[2])
            else:
                print("Invalid signal type.")
                return None

            data.append((index, amplitude))

        return data

    except Exception as e:
        print("Error reading the file:", e)
        return None
def add_signals(data1, data2):
    if len(data1) != len(data2):
        print("Error: The signals have different lengths.")
        return None

    summed_signal = [(sample1[0], sample1[1] + sample2[1]) for sample1, sample2 in zip(data1, data2)]
    return summed_signal
def sub_signals(data1, data2):
    if len(data1) != len(data2):
        print("Error: The signals have different lengths.")
        return None

    subtracted_signal = [(sample1[0], sample1[1] - sample2[1]) for sample1, sample2 in zip(data1, data2)]
    return subtracted_signal
def multiply_signal(data, constant):
    if constant == -1:
        # Invert the signal
        multiplied_signal = [(sample[0], -sample[1]) for sample in data]
    else:
        # Multiply the signal by the constant
        multiplied_signal = [(sample[0], sample[1] * constant) for sample in data]
    return multiplied_signal
def square_signal(data):
    squared_signal = [(sample[0], sample[1] ** 2) for sample in data]
    return squared_signal
def shift_signal(data, constant):
    shifted_signal = [(sample[0] + constant, sample[1]) for sample in data]
    return shifted_signal
def normalize_signal(data, normalize_type):
    if normalize_type == 0:  # Normalize to [0, 1]
        max_amplitude = max(sample[1] for sample in data)
        min_amplitude = min(sample[1] for sample in data)

        if max_amplitude == min_amplitude:
            return data  # Avoid division by zero

        normalized_signal = [(sample[0], (sample[1] - min_amplitude) / (max_amplitude - min_amplitude)) for sample in data]
    elif normalize_type == 1:  # Normalize to [-1, 1]
        max_amplitude = max(sample[1] for sample in data)
        min_amplitude = min(sample[1] for sample in data)

        if max_amplitude == min_amplitude:
            return data  # Avoid division by zero

        normalized_signal = [(sample[0], 2 * (sample[1] - min_amplitude) / (max_amplitude - min_amplitude) - 1) for sample in data]
    else:
        return data

    return normalized_signal
def accumulate_signal(data):
    accumulated_signal = []
    accumulated_amplitude = 0

    for sample in data:
        accumulated_amplitude += sample[1]
        accumulated_signal.append((sample[0], accumulated_amplitude))

    return accumulated_signal
def plot_signal_2(data):
    if len(data) == 0:
        print("No data to plot.")
        return

    index = [sample[0] for sample in data]
    amplitude = [sample[1] for sample in data]

    plt.figure(figsize=(10, 5))

    # Continuous representation
    plt.subplot(1, 2, 1)
    plt.plot(index, amplitude, label="Continuous Signal", marker='o')
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Continuous Signal Representation")
    plt.legend()

    # Discrete representation
    plt.subplot(1, 2, 2)
    plt.stem(index, amplitude, linefmt='-b', markerfmt='ob', basefmt=' ', label="Discrete Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Discrete Signal Representation")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
def Task2():
    # main window
    root = tk.Tk()
    root.geometry("800x500")
    root.title("DSP Task 2")

    header_label = ttk.Label(root, text="DSP Task 2", font=("Arial", 20))
    header_label.pack(pady=20)

    selected_files = {}  # Dictionary to store the selected file labels

    data1 = []  # List to store data for signal 1
    data2 = []  # List to store data for signal 2

    upload_button1 = ttk.Button(root, text="Upload Text File for Signal 1", command=lambda: browse_file_2(1))
    upload_button1.pack()

    selected_file_label1 = ttk.Label(root, text="Selected File 1: ")
    selected_file_label1.pack(pady=10)
    selected_files[1] = selected_file_label1  # Store the label in the dictionary

    upload_button2 = ttk.Button(root, text="Upload Text File for Signal 2", command=lambda: browse_file_2(2))
    upload_button2.pack()

    selected_file_label2 = ttk.Label(root, text="Selected File 2: ")
    selected_file_label2.pack(pady=10)
    selected_files[2] = selected_file_label2  # Store the label in the dictionary

    add_button = ttk.Button(root, text="Add Signals", command=lambda: plot_signal_2(add_signals(data1, data2)))
    add_button.pack()

    sub_button = ttk.Button(root, text="Subtract Signals", command=lambda: plot_signal_2(sub_signals(data1, data2)))
    sub_button.pack()

    const_label = ttk.Label(root, text="Enter A Constant value to use it in Shifting or Multiplying :")
    const_label.pack()

    constant_value = tk.DoubleVar(value=1.0)
    constant_entry = ttk.Entry(root, textvariable=constant_value, width=10)
    constant_entry.pack()

    mul_button = ttk.Button(root, text="Multiply Signal by Constant",command=lambda: plot_signal_2(multiply_signal(data1, constant_value.get())))
    mul_button.pack()

    shift_button = ttk.Button(root, text="Shift Signal", command=lambda: plot_signal_2(shift_signal(data1, constant_value.get())))
    shift_button.pack()

    squaring_button = ttk.Button(root, text="Squaring Signal",command=lambda: plot_signal_2(square_signal(data1)))
    squaring_button.pack()

    normalize_type_var = tk.IntVar(value=0)  # Default: Normalize to [0, 1]

    normalize_label = ttk.Label(root, text="Normalization Type:")
    normalize_label.pack()

    normalize_radiobutton0 = ttk.Radiobutton(root, text="0 to 1", variable=normalize_type_var, value=0)
    normalize_radiobutton1 = ttk.Radiobutton(root, text="-1 to 1", variable=normalize_type_var, value=1)

    normalize_radiobutton0.pack()
    normalize_radiobutton1.pack()

    normalize_button = ttk.Button(root, text="Normalize Signal", command=lambda: plot_signal_2(normalize_signal(data1, normalize_type_var.get())))
    normalize_button.pack()


    accumulate_button = ttk.Button(root, text="Accumulate Signal", command=lambda: plot_signal_2(accumulate_signal(data1)))
    accumulate_button.pack()


    root.mainloop()
#*******************************************************************
#***************************** TASK 3 ******************************
def browse_file_3():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        selected_file_label["text"] = file_path
def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("QuantizationTest1 Test case passed successfully")
def read_signal_data_3(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.read().splitlines()

        if len(lines) < 3:
            print("Invalid file format.")
            return None

        num_samples = int(lines[2])
        data = []

        for i in range(3, 3 + num_samples):
            parts = lines[i].split()
            if len(parts) != 2:
                print("Invalid line format.")
                return None
            index, amplitude = float(parts[0]), float(parts[1])
            data.append((index, amplitude))

        return data

    except Exception as e:
        print("Error reading the file:", e)
        return None
def quantize_signal_3(data, levels=None, num_bits=None): 
    if levels is None and num_bits is None:
        print("Please provide either levels or num_bits.")
        return None, None, None

    if levels is None:
        levels = 2 ** num_bits

    max_amplitude = max(sample[1] for sample in data)
    min_amplitude = min(sample[1] for sample in data)

    amplitude_range = max_amplitude - min_amplitude
    quantization_step = amplitude_range / levels

    quantized_signal = [
        (sample[0], min_amplitude + quantization_step * round((sample[1] - min_amplitude) / quantization_step)) for
        sample in data]
    quantization_error = [(sample[0], sample[1] - quantized_sample[1]) for sample, quantized_sample in
                          zip(data, quantized_signal)]

    return quantized_signal, quantization_error, levels
def quantize_button_3():
    file_path = selected_file_label["text"]
    data = read_signal_data_3(file_path)
    if data is not None:
        quantization_type = quantization_type_var.get()

        if quantization_type == 'levels':
            levels = int(levels_entry.get())
            num_bits = None
        elif quantization_type == 'num_bits':
            num_bits = int(num_bits_entry.get())
            levels = None
        else:
            print("Invalid quantization type.")
            return

        quantized_signal, quantization_error, levels = quantize_signal_3(data, levels, num_bits)

        if quantized_signal is not None and quantization_error is not None:
            plt.figure(figsize=(12, 5))

            # Plot encoded signal
            plt.subplot(1, 3, 1)
            index = [sample[0] for sample in data]
            amplitude = [sample[1] for sample in data]
            plt.plot(index, amplitude, label="Encoded Signal", marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("Encoded Signal")
            plt.legend()

            plt.subplot(1, 3, 2)
            # Plot quantized signal
            index = [sample[0] for sample in quantized_signal]
            amplitude = [sample[1] for sample in quantized_signal]
            plt.plot(index, amplitude, label="Quantized Signal", marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("Quantized Signal")
            plt.legend()

            plt.subplot(1, 3, 3)
            # Plot quantization error
            index = [sample[0] for sample in quantization_error]
            amplitude = [sample[1] for sample in quantization_error]
            plt.plot(index, amplitude, label="Quantization Error", marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("Quantization Error")
            plt.legend()

            plt.tight_layout()
            plt.show()
            # Return the encoded and quantized signals
            encoded_values = [sample[1] for sample in data]
            quantized_values = [sample[1] for sample in quantized_signal]

            # Display encoded and quantized values
            encoded_label["text"] = "Encoded Values: " + str(encoded_values)
            quantized_label["text"] = "Quantized Values: " + str(quantized_values)

            levels_label["text"] = "Levels: " + str(levels)
def Task3():
    # Create the main window
    root = tk.Tk()
    root.geometry("800x500")
    root.title("Signal Quantization")

    # Create and configure widgets
    header_label = ttk.Label(root, text="Signal Quantization", font=("Arial", 20))
    header_label.pack(pady=20)

    upload_button = ttk.Button(root, text="Browse Signal File", command=browse_file_3)
    upload_button.pack()

    selected_file_label = ttk.Label(root)
    selected_file_label.pack(pady=10)

    quantization_type_label = ttk.Label(root, text="Select Quantization Type:")
    quantization_type_label.pack()

    quantization_type_var = tk.StringVar()
    quantization_type_var.set("levels")
    quantization_radiobutton_levels = ttk.Radiobutton(root, text="Levels", variable=quantization_type_var, value="levels")
    quantization_radiobutton_num_bits = ttk.Radiobutton(root, text="Number of Bits", variable=quantization_type_var,
                                                        value="num_bits")
    quantization_radiobutton_levels.pack()
    quantization_radiobutton_num_bits.pack()

    levels_label = ttk.Label(root, text="Levels:")
    levels_label.pack()

    levels_entry = ttk.Entry(root)
    levels_entry.pack()

    num_bits_label = ttk.Label(root, text="Number of Bits:")
    num_bits_label.pack()

    num_bits_entry = ttk.Entry(root)
    num_bits_entry.pack()

    quantize_button = ttk.Button(root, text="Quantize Signal", command=quantize_button_3)
    quantize_button.pack()
    # Labels to display encoded and quantized values
    encoded_label = ttk.Label(root, text="Encoded Values:")
    encoded_label.pack()
    quantized_label = ttk.Label(root, text="Quantized Values:")
    quantized_label.pack()


    # Run the GUI main loop
    root.mainloop()
#*******************************************************************

#***************************** TASK 4 ******************************
#*******************************************************************

#***************************** TASK 5 ******************************
def Task5():
    def SignalSamplesAreEqual(file_name,samples):
        """
        this function takes two inputs the file that has the expected results and your results.
        file_name : this parameter corresponds to the file path that has the expected output
        samples: this parameter corresponds to your results
        return: this function returns Test case passed successfully if your results is similar to the expected output.
        """
        expected_indices=[]
        expected_samples=[]
        with open(file_name, 'r') as f:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            while line:
                # process line
                L=line.strip()
                if len(L.split(' '))==2:
                    L=line.split(' ')
                    V1=int(L[0])
                    V2=float(L[1])
                    expected_indices.append(V1)
                    expected_samples.append(V2)
                    line = f.readline()
                else:
                    break
                    
        if len(expected_samples)!=len(samples):
            print("Test case failed, your signal have different length from the expected one")
            return
        for i in range(len(expected_samples)):
            if abs(samples[i] - expected_samples[i]) < 0.01:
                continue
            else:
                print("Test case failed, your signal have different values from the expected one") 
                return
        print("Test case passed successfully")


    class Task5:
        def __init__(self, root):
            self.root = root
            self.root.title("DCT Calculator")
            self.root.geometry("800x500")
            self.create_widgets()
        num_dct = tk.DoubleVar(value=1.0)

        def create_widgets(self):
            self.label = tk.Label(self.root, text="Upload a text file:")
            self.label.pack()

            self.upload_button = tk.Button(self.root, text="Upload File", command=self.upload_file)
            self.upload_button.pack()

            self.compute_button = tk.Button(self.root, text="Compute DCT", command=self.compute_dct)
            self.compute_button.pack()

            self.save_button = tk.Button(self.root, text="Save Coefficients", command=self.save_coefficients)
            self.save_button.pack()

            num_dct = tk.DoubleVar(value=1.0)
            self.num_dct_entry = ttk.Entry(root, textvariable=num_dct, width=10)
            self.num_dct_entry.pack()

            self.remove_button = tk.Button(self.root, text="Remove DCT", command=self.remove_dct)
            self.remove_button.pack()

            self.result_label = tk.Label(self.root, text="")
            self.result_label.pack()

            self.file_content = ""

        def upload_file(self):
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    self.file_content = file.read()
                self.result_label.config(text="File uploaded successfully.")

        def compute_dct(self):
            if not self.file_content:
                self.result_label.config(text="Please upload a file first.")
                return

            input_data = self.file_content.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
            signal_samples = np.array(values)

            N = len(signal_samples)
            self.dct_coefficients = np.zeros(N)

            for k in range(N):
                for n in range(N):
                    cos_term = np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
                    self.dct_coefficients[k] += signal_samples[n] * cos_term
            self.dct_coefficients *= np.sqrt(2 / N)  
 

            # print("DCT Coefficients:", dct_coefficients)
            
            ###############################################################################
            #         # Load expected coefficients from the file
            #         expected_file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 5\\Task files\\DCT\\DCT_output.txt"
            #         with open(expected_file_name, 'r') as f:
            #             expected_data = [line.split() for line in f.read().split('\n') if line.strip()]

            #         expected_values = [float(value) for _, value in expected_data[3:]]  # Skip the first 3 lines

            #         expected_result = np.array(expected_values)

            #         print("Expected DCT Coefficients:", expected_result)
            # #################################################################################
                    # Test the result using the provided function
            expected_file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 5\\Task files\\DCT\\DCT_output.txt"

            SignalSamplesAreEqual(expected_file_name, self.dct_coefficients)

            self.display_result(self.dct_coefficients)


        def remove_dct(self):
            file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\Lab 5\\Task files\Remove DC component\\DC_component_input.txt"
            with open(file_name, 'r') as f:
                data = [line.split() for line in f.read().split('\n') if line.strip()]

            values = [float(value) for _, value in data[3:]]  
            Data = np.array(values)
            sum = 0
            for element in Data:
                sum += element
            average = sum / len(Data)
            result= []
            for element in Data:
                result.append(round((element-average),3))
            # print("Original: ",Data)
            # print("Result: ",result)# CORRECT BUT you need to take just first 3 decimal numbers to get ACCEPTED
            plt.subplot(1, 2, 1)
            plt.plot(Data, marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("Original")

            plt.subplot(1, 2, 2)
            plt.plot(result,marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("After Removing")

            plt.tight_layout()
            plt.show()
            SignalSamplesAreEqual("D:\Studying\Level 4 sem 1\Digital Signal Processing\Labs\Lab 5\Task files\Remove DC component\DC_component_output.txt", result)
            # plt.plot(result,marker = 'o')
            # plt.title("After removing DCT component")
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.show()


        def display_result(self, result):
            plt.plot(result, marker='o')
            plt.title("DCT Result")
            plt.xlabel("Coefficient Index")
            plt.ylabel("Coefficient Value")
            plt.show()

        def save_coefficients(self):
            if not self.file_content:
                self.result_label.config(text="Please upload a file first.")
                return

            # input_data = self.file_content.split('\n')[3:]
            # input_data = [line.split() for line in input_data if line.strip()]

            # values = zip(*[(int(index), float(value)) for index, value in input_data])
            # signal = np.array(values)

            # dct_result = dct(signal, norm='ortho')

            # m = int(input("Enter the number of coefficients to save: "))
            m = int(self.num_dct_entry.get())
            selected_coefficients = self.dct_coefficients[:m]

            # Specify the new file path and name
            file_path = "D:\Studying\Level 4 sem 1\Digital Signal Processing\Labs\Lab 5\Task files\Saved_Coff.txt"

            # Save the array to a new text file
            np.savetxt(file_path, selected_coefficients, fmt='%d')
            print("DCT Coeffecients Saved Successfully")

    if __name__ == "__main__":
        root = tk.Tk()
        app = Task5(root)
        root.mainloop()

#*******************************************************************

#***************************** TASK 6 ******************************
def Task6():
    class SignalsApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Signals Framework")

            self.create_widgets()

        def create_widgets(self):
            # -1- Label and Entry and button for the number of points (for smoothing)
            num_points_label = tk.Label(self.root, text="Enter the number of points for smoothing:")
            num_points_label.pack()

            num_points_entry = tk.Entry(self.root)
            num_points_entry.pack()

            apply_smoothing_button = tk.Button(self.root, text="Apply Smoothing", command=lambda: self.apply_smoothing(num_points_entry.get()))
            apply_smoothing_button.pack()

            # -2- Button to apply sharpening 
            derivatives_button = tk.Button(self.root, text="Sharpening", command=self.DerivativeSignal)
            derivatives_button.pack()

            # -3- Label and Entry for the number of steps to delay or advance
            delay_label = tk.Label(self.root, text="Enter the number of steps to delay or advance:")
            delay_label.pack()

            delay_entry = tk.Entry(self.root)
            delay_entry.pack()

            apply_delay_advance_button = tk.Button(self.root, text="Apply Delay or Advance", command=lambda: self.apply_delay_advance(delay_entry.get()))
            apply_delay_advance_button.pack()

            # -4- Button to apply folding
            apply_folding_button = tk.Button(self.root, text="Apply Folding", command=self.apply_folding)
            apply_folding_button.pack()

            # -5- Label and Entry and button for the number of steps to delay or advance a folded signal
            delay_folded_label = tk.Label(self.root, text="Enter the number of steps to delay or advance a folded signal:")
            delay_folded_label.pack()

            delay_folded_entry = tk.Entry(self.root)
            delay_folded_entry.pack()

            apply_delay_advance_folded_button = tk.Button(self.root, text="Apply Delay or Advance to Folded Signal", command=lambda: self.apply_delay_advance_folded(delay_folded_entry.get()))
            apply_delay_advance_folded_button.pack()

            # -6- Button to remove DC component in frequency domain
            remove_dc_component_button = tk.Button(self.root, text="Remove DC Component in Frequency Domain", command=self.remove_dc_component)
            remove_dc_component_button.pack()

            # -7- Button to apply convolution
            apply_convolution_button = tk.Button(self.root, text="Apply Convolution", command=self.apply_convolution)
            apply_convolution_button.pack()  
            self.file_content = ""  

        def apply_smoothing(self, num_points):
            try:
                num_points = int(num_points)
            except ValueError:
                # Handle invalid input
                tk.messagebox.showerror("Error", "Invalid input. Please enter a valid integer.")
                return

            # Example signal
            x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

            # Apply moving average for smoothing
            y = self.smooth_signal(x, num_points)

            # Plot the original and smoothed signals
            plt.plot(x, label='Original Signal')
            plt.plot(y, label=f'Smoothed Signal (Moving Average, {num_points} points)')
            plt.legend()
            plt.show()

        def DerivativeSignal(self):# DONE
            
            InputSignal = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
            expectedOutput_first = [1] * (len(InputSignal)-1)
            expectedOutput_second = [0] * (len(InputSignal)-2)

            
            FirstDrev=np.diff(InputSignal)
            SecondDrev=np.diff(FirstDrev)

            """
            End
            """
            
            """
            Testing your Code
            """
            if (len(FirstDrev) != len(expectedOutput_first)) or (len(SecondDrev) != len(expectedOutput_second)):
                print("mismatch in length") 
                print(len(FirstDrev), len(expectedOutput_first))
                print(len(SecondDrev), len(expectedOutput_second))

                return

            first=second=True
            for i in range(len(expectedOutput_first)):
                if abs(FirstDrev[i] - expectedOutput_first[i]) < 0.01:
                    continue
                else:
                    first=False
                    print("1st derivative wrong")
                    return
            for i in range(len(expectedOutput_second)):
                if abs(SecondDrev[i] - expectedOutput_second[i]) < 0.01:
                    continue
                else:
                    second=False
                    print("2nd derivative wrong") 
                    return
            if(first and second):
                print("Derivative Test case passed successfully")
            else:
                print("Derivative Test case failed")
            return

        def apply_delay_advance(self, num_steps):
            try:
                num_steps = int(num_steps)
            except ValueError:
                # Handle invalid input
                tk.messagebox.showerror("Error", "Invalid input. Please enter a valid integer.")
                return

            ######################################################################
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    self.file_content = file.read()
            input_data = self.file_content.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
            ind = np.array(indices)
            x = np.array(values)
            ######################################################################

            # Example signal
            # x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

            # Apply delay or advance
            if num_steps > 0:
                y = np.concatenate((np.zeros(num_steps), x[:-num_steps]))
            elif num_steps < 0:
                y = np.concatenate((x[-num_steps:], np.zeros(-num_steps)))
            else:
                y = x
            ###########
            #Testing
            ###########
            # Plot the original and delayed/advanced signals
            plt.plot(x, label='Original Signal')
            plt.plot(y, label=f'Delayed/Advanced Signal ({num_steps} steps)')
            plt.legend()
            plt.show()

        def apply_folding(self):# DONE
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    self.file_content = file.read()
            input_data = self.file_content.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
            ind = np.array(indices)
            x = np.array(values)

            # Apply folding
            y = np.flip(x)
            outputFile = "D:\Studying\Level 4 sem 1\Digital Signal Processing\Labs\Lab 6\TestCases\Shifting and Folding\Output_fold.txt"
            SFS.Shift_Fold_Signal(outputFile,ind,y)
            # Plot the original and folded signals
            plt.plot(x, label='Original Signal')
            plt.plot(y, label='Folded Signal')
            plt.legend()
            plt.show()

        def apply_delay_advance_folded(self, num_steps):# DONE
            try:
                num_steps = int(num_steps)
            except ValueError:
                # Handle invalid input
                tk.messagebox.showerror("Error", "Invalid input. Please enter a valid integer.")
                return

            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    self.file_content = file.read()
            input_data = self.file_content.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
            ind = np.array(indices)
            x = np.array(values)

            # Apply folding
            y = np.flip(x)

            # Apply delay or advance to the folded signal
            ind += num_steps
            outputFile = "D:\Studying\Level 4 sem 1\Digital Signal Processing\Labs\Lab 6\TestCases\Shifting and Folding\Output_ShifFoldedby500.txt"
            SFS.Shift_Fold_Signal(outputFile,ind,y)
            # Plot the original, folded, and delayed/advanced folded signals
            plt.plot(x, label='Original Signal')
            plt.plot(np.flip(x), label='Folded Signal')
            plt.plot(y, label=f'Delayed/Advanced Folded Signal ({num_steps} steps)')
            plt.legend()
            plt.show()

        def apply_convolution(self):# DONE
            # Provided inputs
            InputIndicesSignal1 = [-2, -1, 0, 1]
            InputSamplesSignal1 = [1, 2, 1, 1]

            InputIndicesSignal2 = [0, 1, 2, 3, 4, 5]
            InputSamplesSignal2 = [1, -1, 0, 0, 1, 1]

            # Perform convolution
            y = np.convolve(InputSamplesSignal1, InputSamplesSignal2, mode='full')
            conv_indices = np.arange(len(y)) + min(InputIndicesSignal1[0], InputIndicesSignal2[0])
            CnvTest.ConvTest(conv_indices,y)
            # Plot the original and convolved signals
            plt.stem(InputIndicesSignal1, InputSamplesSignal1, label='Input Signal 1')
            plt.stem(InputIndicesSignal2, InputSamplesSignal2, label='Input Signal 2', markerfmt='rx')
            plt.stem(conv_indices, y, label='Convolved Signal', markerfmt='go')
            plt.legend()
            plt.show()
        def SignalSamplesAreEqual(self,file_name,samples):
            """
            this function takes two inputs the file that has the expected results and your results.
            file_name : this parameter corresponds to the file path that has the expected output
            samples: this parameter corresponds to your results
            return: this function returns Test case passed successfully if your results is similar to the expected output.
            """
            expected_indices=[]
            expected_samples=[]
            with open(file_name, 'r') as f:
                line = f.readline()
                line = f.readline()
                line = f.readline()
                line = f.readline()
                while line:
                    # process line
                    L=line.strip()
                    if len(L.split(' '))==2:
                        L=line.split(' ')
                        V1=int(L[0])
                        V2=float(L[1])
                        expected_indices.append(V1)
                        expected_samples.append(V2)
                        line = f.readline()
                    else:
                        break
                        
            if len(expected_samples)!=len(samples):
                print("Test case failed, your signal have different length from the expected one")
                return
            for i in range(len(expected_samples)):
                if abs(samples[i] - expected_samples[i]) < 0.01:
                    continue
                else:
                    print("Test case failed, your signal have different values from the expected one") 
                    return
            print("Test case passed successfully")

        def remove_dc_component(self):# DONE
            file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\Lab 5\\Task files\Remove DC component\\DC_component_input.txt"
            with open(file_name, 'r') as f:
                data = [line.split() for line in f.read().split('\n') if line.strip()]

            values = [float(value) for _, value in data[3:]]  
            Data = np.array(values)
            sum = 0
            for element in Data:
                sum += element
            average = sum / len(Data)
            result= []
            for element in Data:
                result.append(round((element-average),3))
            # print("Original: ",Data)
            # print("Result: ",result)# CORRECT BUT you need to take just first 3 decimal numbers to get ACCEPTED
            plt.subplot(1, 2, 1)
            plt.plot(Data, marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("Original")

            plt.subplot(1, 2, 2)
            plt.plot(result,marker='o')
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.title("After Removing")

            plt.tight_layout()
            plt.show()
            self.SignalSamplesAreEqual("D:\Studying\Level 4 sem 1\Digital Signal Processing\Labs\Lab 5\Task files\Remove DC component\DC_component_output.txt", result)
            
            # # Example signal
            # x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

            # # Compute the Discrete Fourier Transform (DFT)
            # X = fft(x)

            # # Set the DC component to zero
            # X[0] = 0

            # # Compute the Inverse Fourier Transform
            # y = ifft(X)

            # # Plot the original and DC component removed signals
            # plt.plot(x, label='Original Signal')
            # plt.plot(np.real(y), label='Signal with DC Component Removed')
            # plt.legend()
            # plt.show()

        def smooth_signal(self, signal, num_points):
            # Apply moving average for smoothing
            smoothed_signal = np.convolve(signal, np.ones(num_points) / num_points, mode='valid')
            return smoothed_signal

    if __name__ == "__main__":
        root = tk.Tk()
        app = SignalsApp(root)
        root.mainloop()

#*******************************************************************
        

#***************************** TASK 7 ******************************
def Task7():
    class CrossCorrelationApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Cross-Correlation App")

            # self.signal1_label = ttk.Label(root, text="Signal 1:")
            # self.signal1_label.grid(row=0, column=0, padx=10, pady=10)

            # self.signal1_entry = ttk.Entry(root)
            # self.signal1_entry.grid(row=0, column=1, padx=10, pady=10)

            # self.signal2_label = ttk.Label(root, text="Signal 2:")
            # self.signal2_label.grid(row=1, column=0, padx=10, pady=10)

            # self.signal2_entry = ttk.Entry(root)
            # self.signal2_entry.grid(row=1, column=1, padx=10, pady=10)

            self.calculate_button = ttk.Button(root, text="Calculate Correlation", command=self.calculate_correlation_and_time_delay)
            self.calculate_button.grid(row=1, column=0, columnspan=2, pady=10)
            # Listbox widgets for displaying correlation and normalized correlation arrays
            self.correlation_listbox = tk.Listbox(root, selectmode=tk.SINGLE)
            self.correlation_listbox.grid(row=3, column=0, padx=10, pady=10)
            self.correlation_listbox_label = ttk.Label(root, text="Correlation Array:")
            self.correlation_listbox_label.grid(row=2, column=0, padx=10, pady=10)

            self.norm_correlation_listbox = tk.Listbox(root, selectmode=tk.SINGLE)
            self.norm_correlation_listbox.grid(row=3, column=1, padx=10, pady=10)
            self.norm_correlation_listbox_label = ttk.Label(root, text="Normalized Correlation Array:")
            self.norm_correlation_listbox_label.grid(row=2, column=1, padx=10, pady=10)
            self.sampling_period_label = ttk.Label(root, text="Sampling Period:")
            self.sampling_period_label.grid(row=4, column=0, padx=10, pady=10)



            self.calculate_button = ttk.Button(root, text="Calculate Time Delay", command=self.calclute_time_delay)
            self.calculate_button.grid(row=6, column=0, columnspan=2, pady=10)
            self.sampling_period_entry = ttk.Entry(root)
            self.sampling_period_entry.grid(row=4, column=1, padx=10, pady=10)

            
            # Entry widget for displaying time delay
            self.time_delay_label = ttk.Label(root, text="Time Delay:")
            self.time_delay_label.grid(row=7, column=0, columnspan=2, padx=10, pady=10)
            self.time_delay_entry = ttk.Entry(root, state='readonly')
            self.time_delay_entry.grid(row=8, column=0, columnspan=2, padx=10, pady=10)

            self.matching_button = ttk.Button(root, text="Template Matching", command=self.template_matching)
            self.matching_button.grid(row=9, column=0, columnspan=2, pady=10)
            # # Matplotlib figure for displaying the signals and correlation result
            # self.figure, self.ax = plt.subplots(3, 1, figsize=(6, 6), tight_layout=True)
            # self.canvas = FigureCanvasTkAgg(self.figure, master=root)
            # self.canvas_widget = self.canvas.get_tk_widget()
            # self.canvas_widget.grid(row=4, column=0, columnspan=2, padx=10, pady=10)
        def calclute_time_delay(self):
            # test = [int(value) for value in values_str if value]

            fs = float(self.sampling_period_entry.get())
            ts = 1 / fs
            signal1= []
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    td1 = file.read()
            input_data = td1.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            # indices1, signal1 = zip(*[(int(index), float(value)) for index, value in input_data])
            for i in range(len(input_data)):
                signal1.append(float(input_data[i][1]))
            # ind1 = np.array(indices1)
            # signal1 = np.array(values)

            signal2 =[]
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    td2 = file.read()
            input_data = td2.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            # indices2, signal2 = zip(*[(int(index), float(value)) for index, value in input_data])
            for i in range(len(input_data)):
                signal2.append(float(input_data[i][1]))
            # ind2 = np.array(indices2)
            # signal2 = np.array(values)

            #############calc correlation####################
            # signal1 = [2, 1, 0, 0, 3]
            # signal2 = [3, 2, 1, 1, 5]

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
                #################################################
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
            # corelation.append(corelation[0])    
            # corelation.remove(corelation[0])    
            # normCorealtion.append(normCorealtion[0])
            # normCorealtion.remove(normCorealtion[0])    
            # Time delay analysis
            max_corr_index = np.argmax(corelation)
            time_delay = max_corr_index * ts
            # Display time delay
            self.time_delay_entry.config(state='normal')
            self.time_delay_entry.delete(0, tk.END)
            self.time_delay_entry.insert(0, f"{time_delay:.4f}")
            self.time_delay_entry.config(state='readonly')
            # print ("helloo")
        def calculate_correlation_and_time_delay(self):
            try:
                signal1 = [2, 1, 0, 0, 3]
                signal2 = [3, 2, 1, 1, 5]

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
                # Time delay analysis
                # max_corr_index = np.argmax(corelation)
                # time_delay = max_corr_index * sampling_period

                # Display signals and correlation result

                plt.subplot(1, 3, 1)
                plt.plot(signal1, marker='o')
                plt.plot(signal2, marker='o')
                plt.title("Original")

                plt.subplot(1, 3, 2)
                plt.plot(corelation,marker='o')
                plt.title("Cross Correlation")

                plt.subplot(1,3,3)
                plt.plot(normCorealtion)
                plt.title("Normalized Cross Correlation")

                plt.tight_layout()
                plt.show()

                # self.ax[0].clear()
                # self.ax[0].plot(signal1, label='Signal 1')
                # self.ax[0].plot(signal2, label='Signal 2')
                # self.ax[0].legend()
                # self.ax[0].set_title('Input Signals')

                # self.ax[1].clear()
                # self.ax[1].plot(corelation, label='Cross-Correlation')
                # self.ax[1].legend()
                # self.ax[1].set_title('Cross-Correlation')

                # self.ax[2].clear()
                # self.ax[2].plot(normCorealtion, label='Normalized Cross-Correlation')
                # self.ax[2].legend()
                # self.ax[2].set_title('Normalized Cross-Correlation')

                # Update Listbox widgets with correlation and normalized correlation arrays
                self.correlation_listbox.delete(0, tk.END)
                self.norm_correlation_listbox.delete(0, tk.END)

                for corr_val, norm_corr_val in zip(corelation, normCorealtion):
                    self.correlation_listbox.insert(tk.END, f"{corr_val:.4f}")
                    self.norm_correlation_listbox.insert(tk.END, f"{norm_corr_val:.4f}")

                # Display time delay
                # self.time_delay_entry.config(state='normal')
                # self.time_delay_entry.delete(0, tk.END)
                # self.time_delay_entry.insert(0, f"{time_delay:.4f}")
                # self.time_delay_entry.config(state='readonly')

                # self.canvas.draw()

            except ValueError as e:
                tk.messagebox.showerror("Error", str(e))
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
            # for i in range(251):
            #     x = 0
            #     y = 0
            #     x = down[0][i] + down[1][i]+ down[2][i] + down[3][i] + down[4][i]
            #     a = x / 5
            #     downAvg.append(a)
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

    root = tk.Tk()
    app = CrossCorrelationApp(root)
    root.mainloop()        
#*******************************************************************
#***************************** TASK 8 ******************************
def Task8():
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

#*******************************************************************
#***************************** TASK 9 ******************************
def Task9():
    class PracticalTask1:

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
            self.M_var = tk.IntVar(value=0)
            self.L_var = tk.IntVar(value=0)
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

            tk.Label(self.master, text="Resampling").grid(row=10, column=0, columnspan=2, pady=10)

            tk.Label(self.master, text="M").grid(row=11, column=0, padx=10, pady=5)
            self.M_entry = tk.Entry(self.master, textvariable=self.M_var, width=25)
            self.M_entry.grid(row=11, column=1, padx=10, pady=5)

            tk.Label(self.master, text="L").grid(row=12, column=0, padx=10, pady=5)
            self.L_entry = tk.Entry(self.master, textvariable=self.L_var, width=25)
            self.L_entry.grid(row=12, column=1, padx=10, pady=5)

            # tk.Label(self.master, text="L").grid(row=5, column=0, padx=10, pady=5)
            # self.fc_entry = tk.Entry(self.master, textvariable=self.f2_var, width=50)
            # self.fc_entry.grid(row=5, column=1, padx=10, pady=5)

            tk.Button(self.master, text="Resample", command=self.resampling).grid(row=13, column=0, columnspan=2, pady=10)

        def upsample(self,signal, factor):
            result = []
            for element in signal:
                result.extend([element] + [0] * (factor-1))
            for i in range(factor-1):
                result.pop()    
            return result

        def downsample(self,signal, factor):
            return signal[::factor]

        def upload_file(self):
            file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            if file_path:
                with open(file_path, 'r') as file:
                    self.file_content = file.read()
            input_data = self.file_content.split('\n')[3:]
            input_data = [line.split() for line in input_data if line.strip()]
            x, y = zip(*[(int(index), float(value)) for index, value in input_data])
            indices = np.array(x)
            values = np.array(y)
            return indices,values

        def resampling(self):
            inputIndecis,input_signal = self.upload_file()
            filteredDataIndcies,filtered_signal = self.run_fir_filter()
            M = (self.M_var.get())
            L = (self.L_var.get())
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
            
            result = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
            test.Compare_Signals(result,indecis,resampled_signal)
            # print(resampled_signal)
            # print(len(resampled_signal))
            # Display the result
            plt.plot(input_signal, label="Original Signal")
            plt.plot(resampled_signal, label="Resampled Signal")
            plt.title("Original vs. Resampled Signal")
            plt.xlabel("Sample Index")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.show()
            # return resampled_signal

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
            
            result, resultIndices, outputFile = self.FIR(filter_type,window,N, FCnormalized,F1Norm,F2Norm)
            self.coefficients = result
            self.coefficientsIndecies = resultIndices
            test.Compare_Signals(outputFile,resultIndices,result)
            self.plot_results(resultIndices, result)
            messagebox.showinfo("NOW","Save your solution")
            self.save_coefficients(resultIndices,result)
            return resultIndices,result

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

    root = tk.Tk()
    app = PracticalTask1(root)
    root.mainloop()
    
#*******************************************************************
        

#***************************** Packege ******************************

# Create the main application window
root = tk.Tk()
root.title("Task Switcher")
root.geometry("200x300")

# Create a variable to hold the currently selected task
current_task = tk.StringVar(value="Task 1")


# Create radio buttons for task selection
task_radio1 = tk.Radiobutton(root, text="Task 1", variable=current_task, value="Task 1", command=Task1)
task_radio2 = tk.Radiobutton(root, text="Task 2", variable=current_task, value="Task 2", command=Task2)
task_radio3 = tk.Radiobutton(root, text="Task 3", variable=current_task, value="Task 3", command=Task3)
task_radio4 = tk.Radiobutton(root, text="Task 4", variable=current_task, value="Task 4", command=Task3)
task_radio5 = tk.Radiobutton(root, text="Task 5", variable=current_task, value="Task 5", command=Task5)
task_radio6 = tk.Radiobutton(root, text="Task 6", variable=current_task, value="Task 6", command=Task6)
task_radio7 = tk.Radiobutton(root, text="Task 7", variable=current_task, value="Task 7", command=Task7)
task_radio8 = tk.Radiobutton(root, text="Task 8", variable=current_task, value="Task 8", command=Task8)
task_radio9 = tk.Radiobutton(root, text="Task 9", variable=current_task, value="Task 9", command=Task9)

# Create a label to display the selected task
task_label = tk.Label(root, text="Select a task:")

# Place the radio buttons and label in the window
task_radio1.pack()
task_radio2.pack()
task_radio3.pack()
task_radio4.pack()
task_radio5.pack()
task_radio6.pack()
task_radio7.pack()
task_radio8.pack()
task_radio9.pack()
task_label.pack()

# Start the main event loop
root.mainloop()
