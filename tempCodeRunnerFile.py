import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
file_content = ""
dct_cof = []
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

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'r') as file:
            file_content = file.read()
        result_label.config(text="File uploaded successfully.")
def compute_dct():
    
    input_data = file_content.split('\n')[3:]
    input_data = [line.split() for line in input_data if line.strip()]
    indices, values = zip(*[(int(index), float(value)) for index, value in input_data])
    signal_samples = np.array(values)

    N = len(signal_samples)
    dct_coefficients = np.zeros(N)

    for k in range(N):
        for n in range(N):
            cos_term = np.cos((np.pi / (4 * N)) * (2 * n - 1) * (2 * k - 1))
            dct_coefficients[k] += signal_samples[n] * cos_term
    dct_coefficients *= np.sqrt(2 / N)  
    expected_file_name = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 5\\Task files\\DCT\\DCT_output.txt"
    dct_cof = dct_coefficients

    SignalSamplesAreEqual(expected_file_name, dct_coefficients)
    display_result(dct_coefficients)
def remove_dct():
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
def display_result(result):
    plt.plot(result, marker='o')
    plt.title("DCT Result")
    plt.xlabel("Coefficient Index")
    plt.ylabel("Coefficient Value")
    plt.show()

def save_coefficients():
    num = int(num_dct_entry.get())
    selected_coefficients = dct_cof[:num]
    file_path = "D:\Studying\Level 4 sem 1\Digital Signal Processing\Labs\Lab 5\Task files\Saved_Coff.txt"
    np.savetxt(file_path, selected_coefficients, fmt='%d')
    print("DCT Coeffecients Saved Successfully")


root = tk.Tk()
root.title("DCT Calculator")
root.geometry("800x500")
file_content = ""
dct_cof=[]
label = tk.Label(root, text="Upload a text file:")
label.pack()

upload_button = tk.Button(root, text="Upload File", command=upload_file)
upload_button.pack()
compute_button = tk.Button(root, text="Compute DCT", command=compute_dct)
compute_button.pack()

num_dct = tk.DoubleVar(value=1)
num_dct_entry = ttk.Entry(root, textvariable=num_dct, width=10)
num_dct_entry.pack()


save_button = tk.Button(root, text="Save Coefficients", command=save_coefficients())
save_button.pack()


remove_button = tk.Button(root, text="Remove DCT", command=remove_dct)
remove_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
