import csv

with open("acronyms_sample.csv", "r") as f:
    reader = csv.reader(f, delimiter=",")
    final_lines = ['acronym,definition']
    for i, line in enumerate(reader):
        if i > 0: # skipp first line
            line_stripped = [l.strip() for l in line if l != '']
            acronym = line_stripped.pop(0)
            for l in line_stripped:
                final_lines.append(f'{acronym},{l}')

with open("acronyms_sample_out.csv", "w") as f:
    for l in final_lines:
        f.write(f'{l}\n')