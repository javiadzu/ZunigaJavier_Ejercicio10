import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap 
from sklearn.model_selection import train_test_split
import sklearn.datasets as skdata
import numpy as np
import pandas as pd
url = "https://github.com/ComputoCienciasUniandes/IntroDataScience/blob/master/ejercicios/10/digitos_primera_parte.ipynb"
# load dataset into Pandas DataFrame
numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
# para poder correr PCA debemos "aplanar las imagenes"
data = imagenes.reshape((n_imagenes, -1)) # para volver a tener los datos como imagen basta hacer data.reshape((n_imagenes, 8, 8))

# Vamos a hacer un split training test
scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.7)

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


  
pca = PCA(n_components = 2) 
  
X_train = pca.fit_transform(x_train) 
X_test = pca.transform(x_test) 
  
explained_variance = pca.explained_variance_ratio_ 




classifier = LogisticRegression(random_state = 0) 
classifier.fit(X_train, y_train) 


# Predicting the test set result using  
# predict function under LogisticRegression  
y_pred = classifier.predict(X_test) 
cm = confusion_matrix(y_test, y_pred) 


X_set, y_set = X_train, y_train 
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                     stop = X_set[:, 0].max() + 1, step = 0.01), 
                     np.arange(start = X_set[:, 1].min() - 1, 
                     stop = X_set[:, 1].max() + 1, step = 0.01)) 
  

  
plt.xlim(X1.min(), X1.max()) 
plt.ylim(X2.min(), X2.max()) 
  
for i, j in enumerate(np.unique(y_set)): 
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], 
                c = ListedColormap(('red', 'green', 'blue','pink','orange','gray','black','magenta','silver','brown','gold'))(i), label = j) 
  
plt.title('Logistic Regression (Training set)') 
plt.xlabel('PC1') # for Xlabel 
plt.ylabel('PC2') # for Ylabel 
plt.legend() # to show legend 
  
# show scatter plot 
plt.show(X_set[i][0],X_set[i][1]) 
#Vamos a crear el mét1odo de clasificación
print(X_set[:,0]**2+X_set[:,1])
print (len(imagenes))
consid= np.zeros(len(imagenes))
for i in range(1255):
    if ((np.array(X_set[i][0]+1)**2)+np.array((X_set[i][1]+4)**2)<2):
        consid[i]=1
    else:
        consid[i]=0

plt.scatter(X_set[:,0],X_set[:,1],consid, cmap=plt.cm.BuPu_r)
plt.xlim(-10,10)
plt.ylim(-10,10)

y_true = np.zeros(len(y_set))
for i in range(len(y_set)):
    if (y_set[i]==1):
        y_true[i] = 1
    else:
        y_true[i] = 0
y_true = y_set
y_pred = consid

confusion_matrix(y_true, y_pred)


