input_file = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/wikielections.txt"
output_file = "/Users/fredericbusch/Desktop/Thesis/forks/py-raccoon/workflow/scripts/datasets/wikielections2.txt"

with open(input_file, mode='r') as infile:
    lines = infile.readlines()

modified_lines = []
for line in lines:
    columns = line.split()
    modified_line = " ".join(columns[:3]) 
    modified_lines.append(modified_line)

with open(output_file, mode='w') as outfile:
    outfile.write("\n".join(modified_lines))

print(f"4th column removed and saved to {output_file}")
