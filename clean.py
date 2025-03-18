import csv

input_file_path = "/Users/yashraj146/Downloads/Demand Forecast/DemoData.csv"
output_file_path = "/Users/yashraj146/Downloads/Demand Forecast/CleanedDemoData.csv"

with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        cleaned_row = [item.replace('"', '') for item in row]
        writer.writerow(cleaned_row)

print(f"Cleaned data saved to {output_file_path}")