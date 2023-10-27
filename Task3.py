import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt

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
def read_signal_data(file_path):
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


def quantize_signal(data, levels=None, num_bits=None):
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



def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
    if file_path:
        selected_file_label["text"] = file_path


def quantize_button():
    file_path = selected_file_label["text"]
    data = read_signal_data(file_path)
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

        quantized_signal, quantization_error, levels = quantize_signal(data, levels, num_bits)

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
# Create the main window
root = tk.Tk()
root.geometry("800x500")
root.title("Signal Quantization")

# Create and configure widgets
header_label = ttk.Label(root, text="Signal Quantization", font=("Arial", 20))
header_label.pack(pady=20)

upload_button = ttk.Button(root, text="Browse Signal File", command=browse_file)
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

quantize_button = ttk.Button(root, text="Quantize Signal", command=quantize_button)
quantize_button.pack()
# Labels to display encoded and quantized values
encoded_label = ttk.Label(root, text="Encoded Values:")
encoded_label.pack()
quantized_label = ttk.Label(root, text="Quantized Values:")
quantized_label.pack()


# Run the GUI main loop
root.mainloop()
