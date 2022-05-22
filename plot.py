import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from datetime import datetime

import warnings 
warnings.filterwarnings("ignore")
# read data
rating_df= pd.read_csv("data/ml-100k/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])

item_df = pd.read_csv("data/ml-100k/ml-100k/u.item", sep="|",encoding="latin-1", 
                      names=["movie_id", "movie_title", "release_date", "video_release_date",
                             "imbd_url", "unknown", "action", "adventure", "animation",
                             "childrens", "comedy", "crime", "documentary", "drama", "fantasy", 
                             "film_noir", "horror", "musical", "mystery", "romance", 
                             "sci-fi", "thriller", "war", "western"])

user_df = pd.read_csv("data/ml-100k/ml-100k/u.user", sep="|", encoding="latin-1", names=["user_id", "age", "gender",
                                                                            "occupation", "zip_code"])
# convert timestamp column to time stamp 
rating_df["timestamp"] = rating_df.timestamp.apply(lambda x: datetime.fromtimestamp(x / 1e3))

# check if change has been applied 
print(rating_df.info())
rating_df.head()

# count the number of male and female raters
gender_counts = user_df.gender.value_counts()
genres= ["unknown", "action", "adventure", "animation", "childrens", "comedy", "crime", "documentary", "drama", "fantasy", "film_noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"]

# plot the counts 
plt.figure(figsize=(9, 5))
plt.bar(x= gender_counts.index[0], height=gender_counts.values[0], color="lightsteelblue")
plt.bar(x= gender_counts.index[1], height=gender_counts.values[1], color="plum")
plt.title("Number of Male and Female Participants", fontsize=16)
plt.xlabel("Gender", fontsize=14)
plt.ylabel("Counts", fontsize=14)
plt.tight_layout()
plt.savefig('gender.png')

full_df = pd.merge(user_df, rating_df, how="left", on="user_id")
full_df = pd.merge(full_df, item_df, how="left", right_on="movie_id", left_on="item_id")
full_df.head()

full_df[genres+["gender"]].groupby("gender").sum().T.plot(kind="barh", figsize=(9,5), color=["lightsteelblue", "plum"])
plt.xlabel("Counts",fontsize=14)
plt.ylabel("Genre", fontsize=14)
plt.title("Popular Genres Among Genders", fontsize=16)
plt.tight_layout()
plt.savefig('gender_cat.png')