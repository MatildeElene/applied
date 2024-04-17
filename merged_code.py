import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.special import softmax

# File paths for each CSV file +  Initialize an empty list to store data + creating category names
file_paths = ['traumas/eating_disorder_validation2.csv', 'traumas/miscarriage_validation2.csv', 'traumas/war_trauma_validation2.csv']
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
count_vectorizer = CountVectorizer(max_df=0.55, min_df=140) #I seem to get the best results at max_df=0.55, min_df=140. 
word_counts = count_vectorizer.fit_transform(combined_df['clean_text'])

# Convert word counts matrix to DataFrame
word_counts_df = pd.DataFrame(word_counts.toarray(), columns=count_vectorizer.get_feature_names_out())

# Filter out columns containing only underscores or non-letter characters
word_counts_df = word_counts_df.loc[:, word_counts_df.columns.str.contains(r'^[a-zA-Z]')]

# Add category column to word counts DataFrame. group by category and sum the word counts
word_counts_df['category'] = combined_df['category']
word_counts_by_category = word_counts_df.groupby('category').sum()
word_counts_by_category.to_csv('roberta_test.csv')  # Has 62,397 columns

#Adding Robertas pipeline
df = pd.read_csv("roberta_test.csv") 
df = df.set_index('category')

norm = df.div(df.sum(axis=1), axis=0) 
soft2 = pd.DataFrame(softmax(norm, axis=0), columns=df.columns, index=df.index)

print(soft2.iloc[0,:].nlargest(20)) #Try adding max_features() instead
