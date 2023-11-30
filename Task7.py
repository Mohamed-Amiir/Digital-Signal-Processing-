import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        self.sampling_period_label = ttk.Label(root, text="Sampling Period:")
        self.sampling_period_label.grid(row=2, column=0, padx=10, pady=10)

        self.sampling_period_entry = ttk.Entry(root)
        self.sampling_period_entry.grid(row=2, column=1, padx=10, pady=10)

        self.calculate_button = ttk.Button(root, text="Calculate", command=self.calculate_correlation_and_time_delay)
        self.calculate_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Listbox widgets for displaying correlation and normalized correlation arrays
        self.correlation_listbox = tk.Listbox(root, selectmode=tk.SINGLE)
        self.correlation_listbox.grid(row=5, column=0, padx=10, pady=10)
        self.correlation_listbox_label = ttk.Label(root, text="Correlation Array:")
        self.correlation_listbox_label.grid(row=4, column=0, padx=10, pady=10)

        self.norm_correlation_listbox = tk.Listbox(root, selectmode=tk.SINGLE)
        self.norm_correlation_listbox.grid(row=5, column=1, padx=10, pady=10)
        self.norm_correlation_listbox_label = ttk.Label(root, text="Normalized Correlation Array:")
        self.norm_correlation_listbox_label.grid(row=4, column=1, padx=10, pady=10)

        # Entry widget for displaying time delay
        self.time_delay_entry = ttk.Entry(root, state='readonly')
        self.time_delay_entry.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
        self.time_delay_label = ttk.Label(root, text="Time Delay:")
        self.time_delay_label.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

        # Matplotlib figure for displaying the signals and correlation result
        self.figure, self.ax = plt.subplots(3, 1, figsize=(6, 6), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def calculate_correlation_and_time_delay(self):
        try:
            signal1 = [2, 1, 0, 0, 3]
            signal2 = [3, 2, 1, 1, 5]
            sampling_period = float(self.sampling_period_entry.get())

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

            # Time delay analysis
            max_corr_index = np.argmax(corelation)
            time_delay = max_corr_index * sampling_period

            # Display signals and correlation result
            self.ax[0].clear()
            self.ax[0].plot(signal1, label='Signal 1')
            self.ax[0].plot(signal2, label='Signal 2')
            self.ax[0].legend()
            self.ax[0].set_title('Input Signals')

            self.ax[1].clear()
            self.ax[1].plot(corelation, label='Cross-Correlation')
            self.ax[1].legend()
            self.ax[1].set_title('Cross-Correlation')

            self.ax[2].clear()
            self.ax[2].plot(normCorealtion, label='Normalized Cross-Correlation')
            self.ax[2].legend()
            self.ax[2].set_title('Normalized Cross-Correlation')

            # Update Listbox widgets with correlation and normalized correlation arrays
            self.correlation_listbox.delete(0, tk.END)
            self.norm_correlation_listbox.delete(0, tk.END)

            for corr_val, norm_corr_val in zip(corelation, normCorealtion):
                self.correlation_listbox.insert(tk.END, f"{corr_val:.4f}")
                self.norm_correlation_listbox.insert(tk.END, f"{norm_corr_val:.4f}")

            # Display time delay
            self.time_delay_entry.config(state='normal')
            self.time_delay_entry.delete(0, tk.END)
            self.time_delay_entry.insert(0, f"{time_delay:.4f}")
            self.time_delay_entry.config(state='readonly')

            self.canvas.draw()

        except ValueError as e:
            tk.messagebox.showerror("Error", str(e))


root = tk.Tk()
app = CrossCorrelationApp(root)