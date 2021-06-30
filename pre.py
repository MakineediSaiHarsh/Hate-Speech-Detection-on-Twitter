import tensorflow as tf
from tensorflow import keras
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import scipy.sparse as sparse
def prediction(df1):
    d = df1.iloc[:,:]
    df1 = df1.iloc[:, 1:]
    df1.replace('[^a-zA-Z]',' ',inplace=True)
    for index in ['text']:
        df1[index]=df1[index].str.lower()

    tf1 = pickle.load(open("tfidf.pkl", 'rb'))
    tfnew=TfidfVectorizer(stop_words='english',vocabulary=tf1.vocabulary_)
    xtt=tfnew.fit_transform(df1['text'])

    def convert_sparse_matrix_to_sparse_tensor(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    xo=convert_sparse_matrix_to_sparse_tensor(xtt)
    xoo=tf.sparse.reorder(xo)

    model=keras.models.load_model("ann.h5")
    pr=model.predict(xoo)
    for i in range(len(pr)):
        pr[i] = round(pr[i][0])
    d['val']=pr
    return d

#top 20
'''tf1 = pickle.load(open("tfidf.pkl", 'rb'))
model=keras.models.load_model("ann.h5")
filename = 'lr_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
c=loaded_model.coef_.tolist()
w=tf1.get_feature_names()
df=pd.DataFrame({'words': w ,'coff':c[0]})
cdf=df.sort_values(['coff','words'], ascending=False)
print(cdf)'''