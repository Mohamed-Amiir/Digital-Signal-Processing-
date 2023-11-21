import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt

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
