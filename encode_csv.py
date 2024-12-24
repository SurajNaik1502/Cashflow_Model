import base64

# Convert the CSV file to Base64
with open("ledger.csv", "rb") as file:
    encoded_csv = base64.b64encode(file.read()).decode('utf-8')

# Save the Base64 string to a text file
with open("encoded_ledger.txt", "w") as output_file:
    output_file.write(encoded_csv)

print("Base64 string has been saved to encoded_ledger.txt")
