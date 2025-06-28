import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv('student_data.csv')
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df['gender'] = df['gender'].map({'M': 0, 'F': 1})
    df = pd.get_dummies(df, columns=['region', 'parent_education'])
    return df

df = load_data()

# Entraîner le modèle
X = df.drop(['student_id', 'dropout'], axis=1)
y = df['dropout']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, y)

# Interface Streamlit
st.title("Prévention de l'abandon scolaire")

# 1. Dashboard Interactif avec des Visualisations
st.header("Analyse exploratoire")
fig, ax = plt.subplots()
sns.histplot(data=df, x='age', hue='dropout', multiple='stack', ax=ax)
st.pyplot(fig)

# 2. Interface de Simulation pour Prédire le Risque d’un Étudiant Fictif
st.header("Prédire le risque d’abandon")
age = st.slider("Âge", 18, 35, 20)
gender = st.selectbox("Genre", ["M", "F"])
region = st.selectbox("Région", ["Urban", "Rural"])
parent_education = st.selectbox("Niveau d’éducation des parents", ["None", "High School", "Bachelor", "Master"])
gpa = st.slider("Moyenne (GPA)", 0.0, 4.0, 2.0)
attendance_rate = st.slider("Taux de présence", 0.0, 1.0, 0.8)
assignments_submitted = st.slider("Devoirs remis", 0, 10, 5)
moodle_time_hours = st.slider("Temps sur Moodle (heures)", 0, 50, 20)
forum_posts = st.slider("Messages sur le forum", 0, 20, 5)
satisfaction_score = st.slider("Score de satisfaction", 1, 5, 3)

# Préparer les données de l’étudiant
input_data = pd.DataFrame({
    'age': [age],
    'gender': [1 if gender == 'F' else 0],
    'gpa': [gpa],
    'attendance_rate': [attendance_rate],
    'assignments_submitted': [assignments_submitted],
    'moodle_time_hours': [moodle_time_hours],
    'forum_posts': [forum_posts],
    'satisfaction_score': [satisfaction_score],
    'region_Urban': [1 if region == 'Urban' else 0],
    'region_Rural': [1 if region == 'Rural' else 0],
    'parent_education_None': [1 if parent_education == 'None' else 0],
    'parent_education_High School': [1 if parent_education == 'High School' else 0],
    'parent_education_Bachelor': [1 if parent_education == 'Bachelor' else 0],
    'parent_education_Master': [1 if parent_education == 'Master' else 0]
})

# S'assurer que input_data a les mêmes colonnes que X dans le même ordre
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Prédiction
input_scaled = scaler.transform(input_data)
proba = rf.predict_proba(input_scaled)[0][1]
st.write(f"**Risque d’abandon : {proba:.2%}**")

# 3. Génération de Recommandations Personnalisées
st.header("Recommandations")
if proba > 0.5:
    st.write("Risque élevé : Suivi personnalisé recommandé.")
    if gpa < 2:
        st.write("Encourager le tutorat académique.")
    if attendance_rate < 0.7:
        st.write("Mettre en place des rappels de présence.")
    if forum_posts < 5:
        st.write("Encourager la participation aux forums.")
else:
    st.write("Risque faible : Continuer à encourager l’engagement.")

# 4. Exportation de Rapports/Exportations en CSV
# Génération et téléchargement du rapport PDF
def generate_pdf():
    pdf_file = "rapport_abandon.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Risque d’abandon : {proba:.2%}", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Âge : {age}", styles['Normal']))
    story.append(Paragraph(f"Genre : {gender}", styles['Normal']))
    story.append(Paragraph(f"Région : {region}", styles['Normal']))
    story.append(Paragraph(f"Moyenne : {gpa}", styles['Normal']))
    doc.build(story)
    return pdf_file

if st.button("Télécharger le rapport PDF"):
    pdf_file = generate_pdf()
    with open(pdf_file, "rb") as f:
        st.download_button("Télécharger", f, file_name=pdf_file)

# Ajout de l'exportation en CSV (nouveau)
if st.button("Télécharger les données en CSV"):
    csv_data = input_data.copy()
    csv_data['risque_abandon'] = proba
    csv_file = "etudiant_simulation.csv"
    csv_data.to_csv(csv_file, index=False)
    with open(csv_file, "rb") as f:
        st.download_button("Télécharger CSV", f, file_name=csv_file) 