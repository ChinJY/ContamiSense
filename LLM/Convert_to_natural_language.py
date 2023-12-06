import pandas as pd

df_train = pd.read_excel(
    io='../../Data files/Exclude/Reformatted/43.xlsx',
    usecols="D:K,X",
    nrows=180
)

df_test = pd.read_excel(
    io='../../Data files/Exclude/Reformatted/43.xlsx',
    usecols="D:K,X",
    skiprows=range(1, 201),
    nrows=3
)

# Iterate through the columns and rows to convert the data
training_data = ""
for index, row in df_train.iterrows():
    for col, value in row.items():
        training_data += f"{col} is {value}"
        if col == 'Contaminated':
            training_data += ". "
        else:
            training_data += ", "

# Remove the trailing comma and space from the last entry
training_data = training_data[:-2]

# Iterate through the columns and rows to convert the data
test_data = []
for index, row in df_test.iterrows():
    new_row = ''
    for col, value in row.items():
        new_row += f"{col} is {value}"
        if col == 'Contaminated':
            new_row += ". "
        else:
            new_row += ", "
    new_row = new_row[:-2] # Remove the trailing comma and space from the last entry
    test_data.append(new_row)

# # Specify the path for the output text file
# output_file = 'output.txt'

# # Open the text file in write mode and save the converted text
# with open(output_file, 'w') as file:
#     file.write(training_data)