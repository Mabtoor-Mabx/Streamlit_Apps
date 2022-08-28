
# Import Libraries
from sklearn.ensemble import RandomForestClassifier
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard.datasets import titanic_survive, feature_descriptions

# Load and Split Dataset

X_train,Y_train,X_test,Y_test = titanic_survive()

# Load Machine Learning Model

model =  RandomForestClassifier().fit(X_train,Y_train)

explainer = ClassifierExplainer(model,
                                X_test,
                                Y_test,
                                cats=['Sex', 'Deck', 'Embarked'],
                                descriptions=feature_descriptions,
                                labels=['not_survived', 'survived']
                               )

ExplainerDashboard(explainer).run()
