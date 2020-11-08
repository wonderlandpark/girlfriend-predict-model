# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
data = pd.read_csv('./data.csv')


# %%
data


# %%
year = data[['year']]
girlfriend = data[['girlfriend']]


# %%
model =  tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,), activation='sigmoid'))


# %%
sgd = tf.keras.optimizers.SGD()


# %%
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# %%
model.fit(np.array(year), np.array(girlfriend), epochs=1000)


# %%
model.predict([30])




