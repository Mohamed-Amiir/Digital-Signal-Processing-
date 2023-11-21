import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

class SignalsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Signals Framework")

        self.create_widgets()

    def create_widgets(self):
        # Label and Entry for the number of points (for smoothing)
        num_points_label = tk.Label(self.root, text="Enter the number of points for smoothing:")
        num_points_label.pack()

        num_points_entry = tk.Entry(self.root)
        num_points_entry.pack()

        # Button to apply smoothing
        apply_smoothing_button = tk.Button(self.root, text="Apply Smoothing", command=lambda: self.apply_smoothing(num_points_entry.get()))
        apply_smoothing_button.pack()

        # Button to apply sharpening (first derivative)
        apply_first_derivative_button = tk.Button(self.root, text="Apply First Derivative", command=self.apply_first_derivative)
        apply_first_derivative_button.pack()

        # Button to apply sharpening (second derivative)
        apply_second_derivative_button = tk.Button(self.root, text="Apply Second Derivative", command=self.apply_second_derivative)
        apply_second_derivative_button.pack()

        # Label and Entry for the number of steps to delay or advance
        delay_label = tk.Label(self.root, text="Enter the number of steps to delay or advance:")
        delay_label.pack()

        delay_entry = tk.Entry(self.root)
        delay_entry.pack()

        # Button to apply delay or advance
        apply_delay_advance_button = tk.Button(self.root, text="Apply Delay or Advance", command=lambda: self.apply_delay_advance(delay_entry.get()))
        apply_delay_advance_button.pack()

        # Button to apply folding
        apply_folding_button = tk.Button(self.root, text="Apply Folding", command=self.apply_folding)
        apply_folding_button.pack()

        # Label and Entry for the number of steps to delay or advance a folded signal
        delay_folded_label = tk.Label(self.root, text="Enter the number of steps to delay or advance a folded signal:")
        delay_folded_label.pack()

        delay_folded_entry = tk.Entry(self.root)
        delay_folded_entry.pack()

        # Button to apply delay or advance to a folded signal
        apply_delay_advance_folded_button = tk.Button(self.root, text="Apply Delay or Advance to Folded Signal", command=lambda: self.apply_delay_advance_folded(delay_folded_entry.get()))
        apply_delay_advance_folded_button.pack()

        # Button to apply convolution
        apply_convolution_button = tk.Button(self.root, text="Apply Convolution", command=self.apply_convolution)
        apply_convolution_button.pack()

        # Button to remove DC component in frequency domain
        remove_dc_component_button = tk.Button(self.root, text="Remove DC Component in Frequency Domain", command=self.remove_dc_component)
        remove_dc_component_button.pack()

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

    def apply_first_derivative(self):
        # Example signal
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Compute the first derivative
        y = np.diff(x)

        # Plot the original and sharpened signals (first derivative)
        plt.plot(x, label='Original Signal')
        plt.plot(y, label='First Derivative')
        plt.legend()
        plt.show()

    def apply_second_derivative(self):
        # Example signal
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Compute the second derivative
        y = np.diff(x, n=2)

        # Plot the original and sharpened signals (second derivative)
        plt.plot(x, label='Original Signal')
        plt.plot(y, label='Second Derivative')
        plt.legend()
        plt.show()

    def apply_delay_advance(self, num_steps):
        try:
            num_steps = int(num_steps)
        except ValueError:
            # Handle invalid input
            tk.messagebox.showerror("Error", "Invalid input. Please enter a valid integer.")
            return

        # Example signal
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Apply delay or advance
        if num_steps > 0:
            y = np.concatenate((np.zeros(num_steps), x[:-num_steps]))
        elif num_steps < 0:
            y = np.concatenate((x[-num_steps:], np.zeros(-num_steps)))
        else:
            y = x

        # Plot the original and delayed/advanced signals
        plt.plot(x, label='Original Signal')
        plt.plot(y, label=f'Delayed/Advanced Signal ({num_steps} steps)')
        plt.legend()
        plt.show()

    def apply_folding(self):
        # Example signal
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Apply folding
        y = np.flip(x)

        # Plot the original and folded signals
        plt.plot(x, label='Original Signal')
        plt.plot(y, label='Folded Signal')
        plt.legend()
        plt.show()

    def apply_delay_advance_folded(self, num_steps):
        try:
            num_steps = int(num_steps)
        except ValueError:
            # Handle invalid input
            tk.messagebox.showerror("Error", "Invalid input. Please enter a valid integer.")
            return

        # Example signal
        x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])

        # Apply folding
        y = np.flip(x)

        # Apply delay or advance to the folded signal
        if num_steps > 0:
            y = np.concatenate((np.zeros(num_steps), y[:-num_steps]))
        elif num_steps < 0:
            y = np.concatenate((y[-num_steps:], np.zeros(-num_steps)))

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
