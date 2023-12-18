# Open the file in read mode
a6 = "D:\\Studying\\Level 4 sem 1\\Digital Signal Processing\\Labs\\Lab 9\\SC, CSYS and DMM\\Practical task 2\\A\\ASeg6.txt"

with open(a6, 'r') as file:
    # Read the lines from the file
    lines = file.readlines()

# Convert the lines to a list of floats
data_array = [float(line.strip()) for line in lines]

# Print or use the array as needed
print(data_array)
