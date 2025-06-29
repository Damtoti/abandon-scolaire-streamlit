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

# Créer une figure avec plusieurs sous-graphiques
plt.figure(figsize=(15, 10))

# 1. Histogramme de l'âge (existante, ajustée)
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='age', hue='dropout', multiple='stack', palette='viridis')
plt.title('Distribution de l’âge par abandon')
plt.xlabel('Âge')
plt.ylabel('Nombre d’étudiants')

# 2. Heatmap de corrélation (existante, ajustée)
plt.subplot(2, 2, 2)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, vmin=-1, vmax=1)
plt.title('Corrélation des variables')

# 3. Boîte à moustaches (Boxplot) pour le GPA par abandon
plt.subplot(2, 2, 3)
sns.boxplot(data=df, x='dropout', y='gpa', palette='Set2')
plt.title('Distribution du GPA par statut d’abandon (0 = Non, 1 = Oui)')
plt.xlabel('Abandon')
plt.ylabel('Moyenne (GPA)')

# 4. Nuage de points (Scatter Plot) entre temps sur Moodle et abandon
plt.subplot(2, 2, 4)
sns.scatterplot(data=df, x='moodle_time_hours', y='attendance_rate', hue='dropout', palette='deep')
plt.title('Temps sur Moodle vs Taux de présence par abandon')
plt.xlabel('Temps sur Moodle (heures)')
plt.ylabel('Taux de présence')

# Ajuster la disposition et sauvegarder
plt.tight_layout()
plt.savefig('eda_visualizations.png')
plt.show()

# 5. Graphique en barres pour le nombre d'abandons par région
plt.figure(figsize=(8, 5))
region_dropout = df.groupby(['region_Urban', 'dropout']).size().unstack(fill_value=0)
region_dropout.plot(kind='bar', stacked=True, color=['#FF9999', '#66B2FF'])
plt.title('Abandons par région (Urban vs Rural)')
plt.xlabel('Région (0 = Rural, 1 = Urban)')
plt.ylabel('Nombre d’étudiants')
plt.legend(['Non abandon', 'Abandon'])
plt.xticks(rotation=0)
plt.savefig('region_dropout.png')
plt.show()