#load dependencies and dataset
!pip install -q seaborn
!pip install -q git+https://github.com/tensorflow/docs

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as pltlines

import tensorflow as tf
from tensorflow import keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

training_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
test_set_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
dataset_cols = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hrs-per-week","native-country","income"]

df = pd.concat([pd.read_csv(training_set_url, names=dataset_cols),pd.read_csv(test_set_url, names=dataset_cols)])
#(train=32561, test=16281) w/o removing unknowns, (train=30162, test=15060) w/ removing unknowns

#cleaning data...
#get rid of some bad string formatting
for col in ["workclass","education","marital-status","occupation","relationship","race","sex","native-country",'income']:
  try:
    df[col] = df[col].str.rstrip('.').str.strip(' ')
  except AttributeError:
    pass

#remove any row that has ? in it as that's equal to NaN in this set
df = df.replace('?',np.NaN)
df = df.dropna()
df.reset_index(inplace=True,drop=True)

### plot initial graphs before preprocessing for the NN ###
# plot the number columns with histogram distributions
custom_red = (0.886,0.29,0.2,1.0)
custom_blue = (0.204,0.541,0.741,1.0)
for col in ['age','fnlwgt','capital-gain','capital-loss','hrs-per-week']:
  plt.style.use('ggplot')
  plt.rcParams.update({
    "figure.facecolor":  (0.0, 0.0, 0.0, 0.0),
    "font.size": 16 
  })
  fig, ax = plt.subplots(figsize=(10,10)) 
  print(col,"vs. Income (Red = under 50K, Blue = over 50K)")
  fig_dist1 = sns.distplot(df.loc[df['income'] == '<=50K'][col], ax=ax, color='red')
  fig_dist2 = sns.distplot(df.loc[df['income'] == '>50K'][col], ax=ax, color=custom_blue)
  
  custom_lines = [pltlines.Line2D([0], [0], color=custom_blue, lw=4),
                pltlines.Line2D([0], [0], color='r', lw=4)]
  plt.legend(custom_lines, ['Over $50K USD', 'Under $50K USD'], fancybox=True)
  plt.show()

  #plot the categorical columns with stacked/scaled bar charts -- done manually so a bit messy
for col in ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']:
  fig, ax = plt.subplots(figsize=(10,10))
  print(col,"vs. Income (Red = under 50K, Blue = over 50K)")
  under_fifty = []
  over_fifty = []
  for item in df[col].unique():
    under_fifty.append([item, len(df.loc[(df['income'] == '<=50K') & (df[col] == item)])/len(df.loc[df[col] == item]) ])
    over_fifty.append([item, len(df.loc[(df['income'] == '>50K') & (df[col] == item)])/len(df.loc[df[col] == item]) ])
  
  u_unzip = [list(e) for e in zip(*under_fifty)]
  u_unzip.append(list(list(zip(*over_fifty))[1]))
  count_df = pd.DataFrame(data=u_unzip)
  count_df = count_df.transpose()
  count_df.columns = ['category','le_50k','gt_50k']
  count_df = count_df.sort_values(by=['gt_50k'],ascending=False)
  count_df.set_index('category',inplace=True)
  
  count_df.plot.bar(ax=ax,stacked=True, color=[custom_red,custom_blue])
  plt.show()

  #NN preprocessing step
#normalize age, fnlwgt, capital-gain, capital-loss, hrs-per-week
for col in ['age','fnlwgt','capital-gain','capital-loss','hrs-per-week']:
  df[col] = df[col].astype(np.float32)/np.max(df[col].astype(np.float32))

#generating one-hots
df = df.join(pd.get_dummies(df['workclass'], prefix='workclass'))
df = df.join(pd.get_dummies(df['education'], prefix='education'))
df = df.join(pd.get_dummies(df['marital-status'], prefix='marital-status'))
df = df.join(pd.get_dummies(df['occupation'], prefix='occupation'))
df = df.join(pd.get_dummies(df['relationship'], prefix='relationship'))
df = df.join(pd.get_dummies(df['race'], prefix='race'))
df = df.join(pd.get_dummies(df['sex'], prefix='sex'))
df = df.join(pd.get_dummies(df['native-country'], prefix='native-country'))
df = df.join(pd.get_dummies(df['income'], prefix='income'))

#drop unneeded columns
df.drop(columns=['income_<=50K','education','education-num','marital-status','occupation','sex','relationship','native-country','race','workclass','income'],inplace=True)

#split the data back into train/test
df_train = df.drop('income_>50K',axis=1)[:30162]
df_train_labels = df['income_>50K'][:30162]
df_test = df.drop('income_>50K',axis=1)[30162:]
df_test_labels = df['income_>50K'][30162:]

scatter_matrix = sns.pairplot(df[["age", "fnlwgt", "capital-gain", "capital-loss","hrs-per-week","income_>50K"]], diag_kind="kde")
plt.show()

#NN Training
model = keras.Sequential([
   keras.layers.Dense(128, activation='relu'),
   keras.layers.Dense(1)
])
model.compile(optimizer='adam',
             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
             metrics=['accuracy'])

history = model.fit(df_train, df_train_labels, epochs=20, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])

#NN Fit Evaluation / Testing
test_loss, test_acc = model.evaluate(df_test,  df_test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

fit_plot = sns.lineplot(hist['epoch'],hist['accuracy'])
plt.legend([pltlines.Line2D([0], [0], color='r', lw=4)],['Fit'])
plt.show()