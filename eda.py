import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les données
df = pd.read_csv('student_data.csv')

# Nettoyage des données
df.fillna(df.mean(numeric_only=True), inplace=True)  # Remplir les valeurs manquantes
df['gender'] = df['gender'].map({'M': 0, 'F': 1})
df = pd.get_dummies(df, columns=['region', 'parent_education'])

# Statistiques descriptives
print(df.describe())

# Visualisations
plt.figure(figsize=(12, 6))

# Histogramme de l'âge
plt.subplot(1, 2, 1)
sns.histplot(data=df, x='age', hue='dropout', multiple='stack')
plt.title('Distribution de l’âge par abandon')

# Heatmap de corrélation
plt.subplot(1, 2, 2)
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Corrélation des variables')

plt.tight_layout()
plt.savefig('eda_visualizations.png')
plt.show()