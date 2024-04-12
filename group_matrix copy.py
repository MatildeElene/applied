import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# File paths for each CSV file
file_paths = ['traumas/eating_disorder_validation2.csv', 'traumas/miscarriage_validation2.csv', 'traumas/war_trauma_validation2.csv']

# Initialize an empty list to store data + creating category names
data = []
new_category_names = ['eating_disorders', 'miscarriage', 'war_trauma']

# Read the "tokenized_text" column from each CSV file and append it to the data list with corresponding category name
for file_path, category_name in zip(file_paths, new_category_names):
    df = pd.read_csv(file_path, usecols=['clean_text'])
    df['category'] = category_name  # Assign new category name
    data.append(df)

# Concatenate the DataFrames in the data list into a single DataFrame
combined_df = pd.concat(data, ignore_index=True)

# Count word frequencies for each category
count_vectorizer = CountVectorizer(min_df=4, stop_words='english')
word_counts = count_vectorizer.fit_transform(combined_df['clean_text'])

# Convert word counts matrix to DataFrame
word_counts_df = pd.DataFrame(word_counts.toarray(), columns=sorted(count_vectorizer.vocabulary_.keys()))

# Filter out columns containing only underscores or non-letter characters
word_counts_df = word_counts_df.loc[:, word_counts_df.columns.str.contains(r'^[a-zA-Z]')]

# Add category column to word counts DataFrame
word_counts_df['category'] = combined_df['category']

# Group by category and sum the word counts
word_counts_by_category = word_counts_df.groupby('category').sum()

# Save the filtered DataFrame to a new CSV file
word_counts_by_category.to_csv('grouped_matrix_raw.csv') #has 171,397 columns (words)