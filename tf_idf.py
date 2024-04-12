import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfTransformer

# Read the word counts by category CSV file into a DataFrame
document_matrix_raw = pd.read_csv('bigger_df_grouped_matrix_raw.csv', index_col=0)

transformer = TfidfTransformer()
output = transformer.fit_transform(document_matrix_raw)

# Get the top 20 values and their corresponding indices for each row
top_values_indices = []
for row in output:
    row = row.toarray().flatten()
    top_indices = row.argsort()[-20:][::-1]  # Get indices of top 20 values
    top_values_indices.append(top_indices)

# Get corresponding words for top indices
word_indices = document_matrix_raw.columns
top_words_per_row = []
for indices in top_values_indices:
    top_words = [word_indices[idx] for idx in indices]
    top_words_per_row.append(top_words)

# Create a DataFrame to display the top 20 words for each row
top_words_df = pd.DataFrame(top_words_per_row, columns=[f"Top_{i+1}" for i in range(20)])

print(top_words_df)