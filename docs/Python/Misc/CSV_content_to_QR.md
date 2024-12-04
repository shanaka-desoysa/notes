---
title: CSV to QR code
date: 2024-12-04
author: Shanaka DeSoysa
description: Create a QR code for content of a CSV file.
---

Here's a Python script that reads a CSV file and generates a QR code from its content:

```python
import qrcode
import csv

# Function to read CSV file and convert its content to a string
def csv_to_string(file_path):
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        csv_data = "\n".join([",".join(row) for row in csv_reader])
    return csv_data

# Path to the CSV file
csv_file_path = 'data.csv'

# Convert CSV content to string
csv_content = csv_to_string(csv_file_path)

# Generate QR code from CSV content
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=4,
)
qr.add_data(csv_content)
qr.make(fit=True)

# Create an image from the QR Code instance
img = qr.make_image(fill='black', back_color='white')

# Save the QR code as "csv_qrcode.png"
img.save("csv_qrcode.png")

print("QR code generated and saved as 'csv_qrcode.png'.")
```

This script reads the content of a CSV file, converts it to a string, and then generates a QR code from that string. The QR code is saved as an image file named `csv_qrcode.png`.

When you run this script, make sure the CSV file (`data.csv`) is in the same directory as your script.
