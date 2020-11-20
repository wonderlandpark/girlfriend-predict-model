# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# get_ipython().run_line_magic('matplotlib', 'inline')
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# %%
data = pd.read_csv('./data.csv')


# %%
year = data[['age']]
girlfriend = data[['girlfriend']]


# %%
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,), activation='sigmoid'))


# %%
sgd = tf.keras.optimizers.SGD()


# %%
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


# %%
model.fit(np.array(year), np.array(girlfriend), epochs=1000)


# %%
model.predict([1])


# %%
x = [ ]
y = [ ]

for i in range(0, 100):
    x.append(i)
    y.append(model.predict([ i ]))


# %%
plt.plot(x, [ a[0] for a in y])
plt.xlabel('Age', fontproperties=fm.FontProperties(fname='C:\\WINDOWS\\Fonts\\Hancom Gothic Regular.ttf', size=10))
plt.ylabel('Girlfriend', fontproperties=fm.FontProperties(fname='C:\\WINDOWS\\Fonts\\Hancom Gothic Regular.ttf', size=10))
plt.title('Girlfirend Predict Model', fontproperties=fm.FontProperties(fname='C:\\WINDOWS\\Fonts\\Hancom Gothic Regular.ttf', size=15))


plt.savefig('graph.png', dpi=300, transparent=False, facecolor='white')


# %%



