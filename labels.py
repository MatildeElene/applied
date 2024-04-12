import pandas as pd
import preprocessing as preprocessing

df = pd.read_pickle("data/data_validation.pkl")

# filter label based on target pos in labels array, clean/preprocess and return dataframe
def filter_and_tokenize(target_df, target_pos_array) -> pd.DataFrame:

    #create empty dict for appending values
    filtered_dict = {
        "work_id": [],
        "labels": [],
        "text": [],
        "clean_text": [],
        "tokenized_text": []
    }

    #loop through df and append rows when target pos matches label value
    for i, row in target_df.iterrows():
        valid = False
        for target_pos in target_pos_array:
            if row["labels"][target_pos] == 1:
                valid = True
        if valid:
            filtered_dict["work_id"].append(row["work_id"])
            filtered_dict["labels"].append(row["labels"])
            filtered_dict["text"].append(row["text"])
            clean_text = preprocessing.preprocess_text(row["text"])
            filtered_dict["clean_text"].append(clean_text)
            filtered_dict["tokenized_text"].append(preprocessing.tokenize_and_lemmatize(clean_text))

    return pd.DataFrame(filtered_dict)


if __name__ == "__main__":
    #e.g. pos 7 = miscarriage
    miscarriage = filter_and_tokenize(df, [7, 20, 23, 25])
    war_trauma = filter_and_tokenize(df, [1, 5, 2, 13, 6, 14])
    eating_disorders = filter_and_tokenize(df, [17, 19, 26])

    miscarriage.to_csv("traumas/miscarriage_validation2.csv")
    war_trauma.to_csv("traumas/war_trauma_validation2.csv")
    eating_disorders.to_csv("traumas/eating_disorder_validation2.csv")
