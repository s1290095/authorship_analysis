# %% [markdown]
# import modules

# %%
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy import sparse

# %% [markdown]
# create data flame from blog corpus

# %%
df = pd.read_csv("input/blogtext.csv")

# %%
df.head()

# %% [markdown]
# Since women tend to express the first person "I" as a lower-case "i," we count the upper-case "I" and lower-case "i" as a feature.(We didn't know what kind of code to write specifically to count the desired words, so I created the following code using gpt.)

# %%
df['i_count'] = df['text'].apply(lambda text: text.count('i'))
df['I_count'] = df['text'].apply(lambda text: text.count('I'))

# %%
df.head()

# %% [markdown]
# Below is a list of expressions that women tend to use.

# %%
female_word = ["adorable", "charming", "sweet", "lovely", "divine", "fabulous", "that was!", ", aren't you?", ", are you?", ", wasn't that?", ", won't you?", ", won't she?", ", did he?", "so much", "Oh !"]

# %% [markdown]
# We count how many of the expressions that women tend to use are used in the text and use them as the feature values.(We didn't know what kind of code to write specifically to count the desired words, so I created the following code using gpt.)

# %%
def female_word_counter(text, female_word):
    cnt = 0
    for w in female_word:
        cnt += text.count(w)
    return cnt

# %%
df['female_word_count'] = df['text'].apply(lambda text: female_word_counter(text, female_word))

# %%
df.head()

# %%
features = df[['i_count', 'I_count', 'female_word_count']]
y = df['gender']

# %% [markdown]
# For machine learning purposes, males and females are represented as 0s and 1s.

# %%
gender_encode = {'male': 0, 'female': 1}
y_encoded = y.map(gender_encode)

# %%
train_x_1, test_x_1, train_y_1, test_y_1 = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

# %%
model1 = LogisticRegression()
model1.fit(train_x_1, train_y_1)

# %% [markdown]
# Use sklearn's CounterVectorizer to extract features from text.

# %%
text = df['text']
text_features = CountVectorizer().fit_transform(text)

# %%
train_x_2, test_x_2, train_y_2, test_y_2 = train_test_split(text_features, y_encoded, test_size=0.2, random_state=42)

# %%
model2 = LogisticRegression()
model2.fit(train_x_2, train_y_2)

# %% [markdown]
# Predict gender by using above 2 models.

# %%
pred1 = model1.predict(test_x_1)

# %%
pred2 = model2.predict(test_x_2)

# %% [markdown]
# Connect above predictions for model stacking.

# %%
stacking_x = pd.concat([pd.Series(pred1), pd.Series(pred2)], axis=1)

# %%
train_x_stacking, test_x_stacking, train_y_stacking, test_y_stacking = train_test_split(stacking_x, test_y_2, test_size=0.2,random_state=42)

# %% [markdown]
# Create finale model.(model stacking)

# %%
stacking_model = LogisticRegression()
stacking_model.fit(train_x_stacking, train_y_stacking)

# %% [markdown]
# Execute finale predictions.

# %%
pred_finale = stacking_model.predict(test_x_stacking)

# %%
accuracy = accuracy_score(test_y_stacking, pred_finale)
print(f"accuracy: {accuracy}")


