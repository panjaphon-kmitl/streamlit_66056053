import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from rfc_func import random_forrest_classifier


# --- Build Model ---
df = pd.read_csv('files/csv/iris.csv')
features = ['sepal.length', 'sepal.width', 'petal.length', 'petal.width']
rfc, uniques, score = random_forrest_classifier(
    df=df,
    x=features,
    y='variety'
)
print('Our accuracy score for this model is {}'.format(score))


# --- Save Pickle Data ---

# Model
rf_pickle = open('files/pickle/random_forest_iris.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

# Unique mapping
output_pickle = open('files/pickle/output_iris.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()


# --- Plotting Feature Importance and Save Image ---
fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features)
plt.title('Which features are the most important for variety prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('files/png/feature_importance.png')