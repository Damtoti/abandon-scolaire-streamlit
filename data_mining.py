import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Charger les données
df = pd.read_csv('student_data.csv')
df.fillna(df.mean(numeric_only=True), inplace=True)
df['gender'] = df['gender'].map({'M': 0, 'F': 1})
df = pd.get_dummies(df, columns=['region', 'parent_education'])

# Séparer les features et la cible
X = df.drop(['student_id', 'dropout'], axis=1)
y = df['dropout']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Clustering (K-Means)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualisation des clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
plt.title('Clusters d’étudiants')
plt.savefig('clusters.png')
plt.show()

# 2. Classification (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print("Précision Random Forest:", rf.score(X_test, y_test))

# Importance des variables
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Importance des variables')
plt.savefig('feature_importance.png')
plt.show()

# 3. Règles d’association
# Binariser les données pour Apriori
df_bin = df[['gpa', 'attendance_rate', 'assignments_submitted', 'dropout']].copy()
df_bin['gpa_low'] = df_bin['gpa'] < 2
df_bin['attendance_low'] = df_bin['attendance_rate'] < 0.7
df_bin['assignments_low'] = df_bin['assignments_submitted'] < 5
df_bin['dropout'] = df_bin['dropout'] == 1

frequent_itemsets = apriori(df_bin[['gpa_low', 'attendance_low', 'assignments_low', 'dropout']], min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
print("Règles d’association:", rules)