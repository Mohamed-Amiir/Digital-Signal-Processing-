import numpy as np
import matplotlib.pyplot as plt

def calculate_LowPass_HD(FCnorm, n):
    if (n == 0):
        result = 2 * FCnorm
    else:    
        result = 2 * FCnorm * ((np.sin(n * 2 * np.pi * FCnorm)) / (n * 2 * np.pi * FCnorm))
    return result

def calculate_HammingW(n, N):
    result = 0.54 + 0.46 * np.cos((2 * np.pi * n) / N)
    return result

def FIR(N, FCnorm):
    H = []
    indices = []
    for n in range(int(-(N // 2)), int((N // 2) + 1)):
        H.append(calculate_LowPass_HD(FCnorm, n) * calculate_HammingW(n, N))
        indices.append(n)
    return H, indices

fc = 1.5
TW = 0.5
As = 50
fs = 8

deltaF = TW / fs
N = 3.3 / deltaF
if int(N) % 2 != 0:
    N = int(N)
else:
    N = int(np.ceil(3.3 / deltaF))
FCnormalized = (fc / fs) + (deltaF / 2)
result, resultIndices = FIR(N, FCnormalized)

plt.plot(resultIndices, result)
plt.title('FIR Lowpass Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.show()
