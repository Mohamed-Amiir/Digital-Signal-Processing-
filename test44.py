import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Frame, IntVar, DoubleVar, Button, Label, Entry, Listbox
# Initialize the input signal
signal = np.array([1, 3, 5, 7, 9, 11, 13, 15])
def calculate_fourier_transform(signal):
    n = len(signal)
    frequency = np.fft.fftfreq(n)
    amplitude_spectrum = np.abs(np.fft.fft(signal))
    phase_spectrum = np.angle(np.fft.fft(signal))

    return frequency, amplitude_spectrum, phase_spectrum

def update_plots(signal):
    frequency, magnitude_spectrum, phase_spectrum = calculate_fourier_transform(signal)

    plt.subplot(2, 1, 1)
    plt.plot(frequency, magnitude_spectrum)
    plt.title('Frequency vs Amplitude')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(frequency, phase_spectrum)
    plt.title('Frequency vs Phase')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (radians)')

    plt.show()

def update_amplitude(index, value):
    signal[index] = signal[index] * value
    update_plots(signal)

def update_phase(index, value):
    signal[index] = signal[index] * np.exp(1j * value)
    update_plots(signal)

def main():
    

    # Calculate the initial Fourier transform
    frequency, magnitude_spectrum, phase_spectrum = calculate_fourier_transform(signal)

    # Create a GUI window
    root = Tk()
    root.title('Fourier Transform Manipulation')

    # Create a frame to display frequency components
    frequency_frame = Frame(root)
    frequency_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Create a listbox to display frequency components
    frequency_listbox = Listbox(frequency_frame, width=20, height=10)
    frequency_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

    # Create labels and entry fields for amplitude and phase modification
    amplitude_label = Label(frequency_frame, text='Amplitude:')
    amplitude_label.pack(side=tk.TOP)
    amplitude_entry = Entry(frequency_frame)
    amplitude_entry.pack(side=tk.TOP)

    phase_label = Label(frequency_frame, text='Phase:')
    phase_label.pack(side=TK.TOP)
    phase_entry = Entry(frequency_frame)
    phase_entry.pack(side=tk.TOP)

    # Create a button to apply modifications
    apply_button = Button(frequency_frame, text='Apply')
    apply_button.pack(side=tk.TOP)

    # Populate the listbox with frequency components
    for i in range(len(frequency)):
        frequency_listbox.insert(i, f'{frequency[i]:.3f}')

    # Update plots and listbox selection on listbox selection
    frequency_listbox.bind('<<ListboxSelect>>', lambda event: update_plots(signal))

    # Update amplitude and phase on entry value changes
    amplitude_entry.bind('<KeyRelease>', lambda event: update_amplitude(frequency_listbox.curselection()[0], float(amplitude_entry.get())))
    phase_entry.bind('<KeyRelease>', lambda event: update_phase(frequency_listbox.curselection()[0], float(phase_entry.get())))

    # Run the GUI event loop
    root.mainloop()

if __name__ == '__main__':
    main()
