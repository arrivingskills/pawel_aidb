import pandas as pd

def compress(input_path, output_path):
    df = pd.read_csv(input_path)
    
    titles_df = df[['title']]
    
    titles_df.to_csv(output_path, index=False)


fake_input = "data/Fake/Fake.csv"
true_input = "data/True/True.csv"

fake_output = "data/Fake/compressedFake.csv"
true_output = "data/True/compressedTrue.csv"

compress(fake_input, fake_output)
compress(true_input, true_output)
