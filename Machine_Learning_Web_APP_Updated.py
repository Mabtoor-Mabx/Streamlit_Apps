
# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from streamlit_embedcode import github_gist

# Header

st.header('**Machine Learning Web App**')
st.markdown('''
**Explore The 3 Different Machine Learning Models with 3 Different Datasets**
''')

# Dataset Names

dataset_name = st.sidebar.selectbox('Choose The Dataset', ('Iris', 'Breast Cancer','Wine'))

# Classifier Name

classifier_name = st.sidebar.selectbox('Choose The Classifier', ('SVM', 'KNN', 'Random Forest'))


# Load The Dataset (For Loading The Dataset, We have to create Custom Function)

def get_data(dataset_name):
    data = None
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

# Call The Get_data Function

x, y = get_data(dataset_name)

# Display Shape and Number Of Classes

st.write('Shape of Dataset :', x.shape )
st.write('Number Of Classes in Dataset :', len(np.unique(y)))

# Parameters of Different Machine Learning Classifiers

def add_parameters(classifier_name):
    parameters = dict()
    if classifier_name == 'SVM':
        C = st.sidebar.slider('c', 0.01,10.0)
        parameters['c'] = C
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K',1,15)
        parameters['K']= K
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        parameters['max_depth']= max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1,100)
        parameters['n_estimators'] = n_estimators
    return parameters

# Call The Add Parameter Slider Function

parameters = add_parameters(classifier_name)


# Create Classifier on the basis of classifier name and parameters

def get_classifier(classifier_name, parameters):
    classifier = None
    if classifier_name == 'SVM':
        classifier = SVC(C=parameters['c'])
    elif classifier_name == 'KNN':
        classifier = KNeighborsClassifier(n_neighbors=parameters['K'])
    else:
        classifier = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                                            max_depth=parameters['max_depth'],
                                           random_state=1234)
    return classifier    

# Call The Get Classifier

classifier = get_classifier(classifier_name, parameters)


# Split The Dataset
X_train, X_test , Y_train,Y_test = train_test_split(x,y, test_size=0.2,random_state=32)

# Train The Classifier

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)


# Check Model Accuracy

accuracy = accuracy_score(Y_test, Y_pred)
st.write(f"Classifier = {classifier_name}")
st.write('Accuracy Score is :', accuracy)

 # Plot The Dataset Using PCA
pca = PCA(2)
X_projected = pca.fit_transform(x)

   ## Slice The Dimension
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure(figsize=(11,3))
fig.patch.set_facecolor('grey')
plt.rcParams['axes.facecolor'] = 'orange'

plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')
plt.xlabel('Principle Component 1', color='white')
plt.ylabel('Principle Component 2', color='white')
plt.colorbar()

# Show The Plot

st.pyplot(fig)

link = 'https://gist.github.com/Mabtoor-Mabx/de3b985ffcba0fffeee68dee2e97246c'
github_gist(link, height=400, width=800)
    
    
  ####-----------------------------------------------------------------------------#######

# Show Code if Someone wants to display code of this app

if st.checkbox('Show Code'):
    with st.echo():
        # Import Libraries
        import streamlit as st
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn import datasets
        from sklearn.model_selection import train_test_split
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        
        
        # Load The Dataset (For Loading The Dataset, We have to create Custom Function)

        def get_data(dataset_name):
            data = None
            if dataset_name == 'Iris':
                data = datasets.load_iris()
            elif dataset_name == 'Breast Cancer':
                data = datasets.load_breast_cancer()
            else:
                data = datasets.load_wine()
            x = data.data
            y = data.target
            return x,y
        
        # Parameters of Different Machine Learning Classifiers

        def add_parameters(classifier_name):
            parameters = dict()
            if classifier_name == 'SVM':
                C = st.sidebar.slider('c', 0.01,10.0)
                parameters['c'] = C
            elif classifier_name == 'KNN':
                K = st.sidebar.slider('K',1,15)
                parameters['K']= K
            else:
                max_depth = st.sidebar.slider('max_depth',2,15)
                parameters['max_depth']= max_depth
                n_estimators = st.sidebar.slider('n_estimators', 1,100)
                parameters['n_estimators'] = n_estimators
            return parameters
        
        
        # Get Classifier Function
        def get_classifier(classifier_name, parameters):
            classifier = None
            if classifier_name == 'SVM':
                classifier = SVC(C=parameters['c'])
            elif classifier_name == 'KNN':
                classifier = KNeighborsClassifier(n_neighbors=parameters['K'])
            else:
                classifier = RandomForestClassifier(n_estimators=parameters['n_estimators'],
                                                    max_depth=parameters['max_depth'],
                                                   random_state=1234)
            return classifier    
        
        
        # Split The Dataset
        X_train, X_test , Y_train,Y_test = train_test_split(x,y, test_size=0.2,random_state=32)

        # Train The Classifier

        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)


        # Check Model Accuracy

        accuracy = accuracy_score(Y_test, Y_pred)
        
        
