a1 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg1.txt"
a2 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg2.txt"
a3 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg3.txt"
a4 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg4.txt"
a5 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg5.txt"
a6 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg6.txt"

b1 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg1.txt"
b2 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg2.txt"
b3 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg3.txt"
b4 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg4.txt"
b5 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg5.txt"
b6 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\B\\BSeg6.txt"
Apath = [a1, a2, a3, a4, a5, a6]
Bpath = [b1, b2, b3, b4, b5, b6]
A = []
B = []

for path in Apath:
    with open(path, 'r') as file:
        lines = file.readlines()
    data_array = [float(line.strip()) for line in lines]
    A.append(data_array)

for path in Bpath:
    with open(path, 'r') as file:
        lines = file.readlines()
    data_array = [float(line.strip()) for line in lines]
    B.append(data_array)


print(A[2])