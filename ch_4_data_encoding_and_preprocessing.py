# %%
sentence = "The President vetoed the bill"
tokenized_sentence = sentence.split(' ')

# %%
vocabulary = {token: i for i, token in enumerate(set(tokenized_sentence))}


# %%
print(vocabulary)

# %%
x = ["The dog is brown",
        "The cat is grey",
        "The dog runs fast",
        "The house is blue"]
y= [1,0,1,0]

# %%
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)
x_vectorized.todense()

# %%
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2 

feature_selector = SelectKBest(k=2)
feature_selector.fit_transform(x_vectorized, y).todense()

# %%
feature_selector.get_support()

# %% 
import numpy as np 
np.array(vectorizer.get_feature_names())[feature_selector.get_support()]

# %%
x = []
y = []
for i in range(1000):
        x.append([0,1])
        y.append(-1)
        x.append([1,0])
        y.append(1)


# %%
from keras.layers import Input, Embedding, Dense, SimpleRNN 
from keras.models import Model 

alphabet_size = 2
embedding_size = 4
sequence_length = 2

input_sequence = Input(shape=(sequence_length,))
embedding = Embedding(alphabet_size,
                        embedding_size,
                        input_length=sequence_length)(input_sequence)
h1 = SimpleRNN(10, return_sequence=False)(embedding)
y_out = Dense(1, activation='linear')(h1)

model = Model(inputs=[input_sequence], outputs=[y_out])
model.compile('RMSProp', loss='mean_squared_error')

# %%
model.fit(x, y, epochs=10)

# %%
from sklearn.linear_model import LinearRegression
import pandas as pd 

x = np.random.normal(size=1000)
y = x + .1*np.random.normal(size=1000)
X = pd.DataFrame({'x': x, 'y': y})

model = LinearRegression()
model.fit(X[['x']], X['y']) 
model.score(X[['x']], X['y'])

# %%
x_discretized = pd.get_dummies(pd.qcut(x, [0., 0.25, .5, .75, 1.])).values[:, 1:]
model.fit(x_discretized, X['y'])
model.score(x_discretized, X['y'])

# %%
