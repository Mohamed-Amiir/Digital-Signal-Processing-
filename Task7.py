import tkinter as tk
from tkinter import filedialog, ttk
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
        self.time_delay_label = ttk.Label(root, text="Time Delay:")
        self.time_delay_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)
        self.time_delay_entry = ttk.Entry(root, state='readonly')
        self.time_delay_entry.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

        self.matching_button = ttk.Button(root, text="Template Matching", command=self.template_matching)
        self.matching_button.grid(row=8, column=0, columnspan=2, pady=10)
        # # Matplotlib figure for displaying the signals and correlation result
        # self.figure, self.ax = plt.subplots(3, 1, figsize=(6, 6), tight_layout=True)
        # self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        # self.canvas_widget = self.canvas.get_tk_widget()
        # self.canvas_widget.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

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
            self.time_delay_entry.config(state='normal')
            self.time_delay_entry.delete(0, tk.END)
            self.time_delay_entry.insert(0, f"{time_delay:.4f}")
            self.time_delay_entry.config(state='readonly')

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