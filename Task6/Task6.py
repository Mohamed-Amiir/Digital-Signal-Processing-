import numpy as np
import matplotlib.pyplot as plt
import comparesignals as cmp
import Shift_Fold_Signal as SFS
from scipy.fft import fft, ifft
import tkinter as tk
from tkinter import filedialog, ttk
import numpy as np
import matplotlib.pyplot as plt
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

    def apply_convolution(self):
        # Example signals
        x = np.array([1, 2, 3, 4, 5])
        h = np.array([0.5, 0.5])

        # Perform convolution
        y = np.convolve(x, h, mode='full')

        # Plot the original and convolved signals
        plt.plot(x, label='Input Signal')
        plt.plot(h, label='Impulse Response')
        plt.plot(y, label='Convolved Signal')
        plt.legend()
        plt.show()

    def remove_dc_component(self):
        # Example signal
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Compute the Discrete Fourier Transform (DFT)
        X = fft(x)

        # Set the DC component to zero
        X[0] = 0

        # Compute the Inverse Fourier Transform
        y = ifft(X)

        # Plot the original and DC component removed signals
        plt.plot(x, label='Original Signal')
        plt.plot(np.real(y), label='Signal with DC Component Removed')
        plt.legend()
        plt.show()

    def smooth_signal(self, signal, num_points):
        # Apply moving average for smoothing
        smoothed_signal = np.convolve(signal, np.ones(num_points) / num_points, mode='valid')
        return smoothed_signal

if __name__ == "__main__":
    root = tk.Tk()
    app = SignalsApp(root)
    root.mainloop()
