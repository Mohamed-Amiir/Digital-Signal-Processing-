number = -1.135854E-19

# Using format function
formatted_number = "{:e}".format(number)
print("Formatted Number (using format):", formatted_number)

# Using f-string (available in Python 3.6 and above)
formatted_number_fstring = f"{number:e}"
print("Formatted Number (using f-string):", formatted_number_fstring)
