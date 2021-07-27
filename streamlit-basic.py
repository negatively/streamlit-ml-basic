from matplotlib.colors import Colormap
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

st.title('Scikit Learn Machine Learning')
st.write('''
Exploring different classifier
''')

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Wine", "Breast Cancer"))

classifier_name = st.sidebar.selectbox('Select Classifier',('KNN', "SVM", "Random Forest"))

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        dataset = datasets.load_iris()
    elif dataset_name == 'Wine':
        dataset = datasets.load_wine()
    elif dataset_name == 'Breast Cancer':
        dataset = datasets.load_breast_cancer()
    X = dataset.data
    y = dataset.target
    return X, y

X, y = get_dataset(dataset_name)
st.write('Shape of Dataset : ', X.shape)
st.write('Number of Classes : ', len(np.unique(y)))

def add_parameter_ui(classifier_name):
    params = {}
    if classifier_name == 'KNN':
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif classifier_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)
        params['C'] = C
    elif classifier_name == 'Random Forest':
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name, params):
    if classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'])
    return clf

clf = get_classifier(classifier_name, params)

# Split Train Test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=43)

# Classification
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier Name : {classifier_name}')
st.write(f'Model Accuracy : {acc}')

# PLOT
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.5, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)

