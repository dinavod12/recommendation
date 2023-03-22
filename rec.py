import pandas as pd
path = "Desktop/archive_website/myntradataset/"
df = pd.read_csv(path + "styles.csv", error_bad_lines=False)
df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
df = df.iloc[0:20000,:]
df.head(10)

df = df[["id","subCategory","articleType","productDisplayName","image"]]
df.head()

df.dropna(inplace  =True)

df["productDisplayName"] = df["productDisplayName"].apply(lambda x :x.split())
df["subCategory"] = df["subCategory"].apply(lambda x :x.split())
df["articleType"] = df["articleType"].apply(lambda x :x.split())

df["tags"] = df["subCategory"]+df["articleType"]+df["productDisplayName"]

df_new = df.drop(columns = ["subCategory","productDisplayName"])

df_new.tail()

df_new["tags"] = df_new["tags"].apply(lambda x : "".join(x))

df_new.head()

df_new["tags"] = df_new["tags"].apply(lambda x:x.lower())

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return "".join(y)

df_new["tags"] = df_new["tags"].apply(stem)
df_new["articleType"] = df_new["articleType"].apply(lambda x : "".join(x))

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000,stop_words="english")

val = cv.fit_transform(df_new["tags"]).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(val)

def recommend(clothes):
    lst = []
    index = df_new[df_new['articleType'] == clothes].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        lst.append(df_new.iloc[i[0]].articleType)
    return lst
        
recommend(input("Product Name"))
