# Applied - Script pipeline 

**Data Collection and Preprocessing**
1. large_json_parser.py #This script reads data from a JSONL file, extracts specific fields, creates a DataFrame, and saves it as a CSV file.

2. preprocessing.py #This script preprocesses text data from a CSV file by removing HTML tags, entities, punctuation, digits, and stopwords, then lemmatizes the tokens, and finally saves the processed DataFrame to a new CSV file.

**Label-Based Filtering and Tokenization**
3. labels.py #This code reads a DataFrame from a pickle file, filters the data based on specific labels, preprocesses the text, tokenizes and lemmatizes it, and then saves the filtered and processed data into separate CSV files based on the target labels.

**TF-IDF Transformation and Softmax Scores**
4. merged_code.py #This script reads preprocessed text data from multiple CSV files, combines them into a single DataFrame, computes word frequencies for each category, applies TF-IDF transformation, computes softmax scores for each word, and outputs the top 20 most relevant words for a specific category.
