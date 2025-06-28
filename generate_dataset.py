import pandas as pd
import numpy as np
from datetime import datetime

np.random.seed(42)
n_students = 1000

# Génération des données
data = {
    'student_id': range(1, n_students + 1),
    'age': np.random.randint(18, 35, n_students),
    'gender': np.random.choice(['M', 'F'], n_students),
    'region': np.random.choice(['Urban', 'Rural'], n_students),
    'parent_education': np.random.choice(['None', 'High School', 'Bachelor', 'Master'], n_students),
    'gpa': np.round(np.random.uniform(0, 4, n_students), 2),
    'attendance_rate': np.random.uniform(0.5, 1, n_students),
    'assignments_submitted': np.random.randint(0, 10, n_students),
    'moodle_time_hours': np.random.uniform(0, 50, n_students),
    'forum_posts': np.random.randint(0, 20, n_students),
    'satisfaction_score': np.random.randint(1, 5, n_students),
    'dropout': np.random.choice([0, 1], n_students, p=[0.8, 0.2])  # 20% d'abandon
}

# Création du DataFrame
df = pd.DataFrame(data)

# Sauvegarde en CSV
df.to_csv('student_data.csv', index=False)
print("Dataset généré et sauvegardé sous 'student_data.csv'")