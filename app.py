
import streamlit as st
import pandas as pd
import pickle
import subprocess
import sys

# Configuração da página
st.set_page_config(page_title="Triagem Interativa - Risco Cardíaco", layout="wide")

# --- FUNÇÃO DE TREINAMENTO ---
def train_model():
    with st.spinner('Iniciando o processo de treinamento...'):
        process = subprocess.Popen(
            [sys.executable, "train_model.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Exibir logs em tempo real
        st_stdout = st.empty()
        st_stderr = st.empty()
        
        while process.poll() is None:
            st_stdout.text(f"stdout: {process.stdout.readline()}")
            st_stderr.text(f"stderr: {process.stderr.readline()}")
        
        # Capturar o restante da saída
        stdout, stderr = process.communicate()
        st_stdout.text(f"stdout: {stdout}")
        st_stderr.text(f"stderr: {stderr}")

        if process.returncode == 0:
            st.success("Modelo treinado com sucesso!")
        else:
            st.error("O treinamento do modelo falhou.")


# --- CARREGAMENTO DOS ARTEFATOS ---
@st.cache_resource
def load_artifacts():
    try:
        with open('artifacts/best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('artifacts/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('artifacts/label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('artifacts/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        return model, scaler, label_encoders, feature_names
    except FileNotFoundError:
        st.error("Artefatos do modelo não encontrados. Treine o modelo primeiro.")
        return None, None, None, None


model, scaler, label_encoders, feature_names = load_artifacts()

# --- FUNÇÃO DE PREDIÇÃO ---
def predict_heart_attack_risk(patient_data, model, scaler, label_encoders, feature_names):
    patient_df = pd.DataFrame([patient_data])
    # Flag de ausência do consumo de álcool, replicando o notebook.
    if 'alcohol_consumption' in patient_df:
        patient_df['alcohol_missing'] = patient_df['alcohol_consumption'].isnull().astype(int)
    else:
        patient_df['alcohol_missing'] = 0

    for col, encoder in label_encoders.items():
        if col in patient_df.columns:
            # Safely transform data, handling unseen labels
            patient_df[col] = patient_df[col].astype(str).apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            patient_df[col] = encoder.transform(patient_df[col].astype(str))
    
    patient_df = patient_df.reindex(columns=feature_names, fill_value=0)
    patient_scaled = scaler.transform(patient_df)
    
    prediction = model.predict(patient_scaled)[0]
    probability = model.predict_proba(patient_scaled)[0]
    
    return {
        'prediction': int(prediction),
        'risk_label': 'ALTO RISCO' if prediction == 1 else 'BAIXO RISCO',
        'probability_no_attack': probability[0] * 100,
        'probability_attack': probability[1] * 100
    }

# --- INTERFACE DO STREAMLIT ---
st.title('🩺 Formulário de Triagem para Risco de Ataque Cardíaco')
st.markdown("---")

# Coletar dados do paciente em abas
patient_data = {}
new_patient_data = {}

main_tabs = st.tabs(["**Triagem de Risco**", "**Cadastrar Novo Paciente**", "**Treinar Modelo**"])

with main_tabs[0]:
    st.header("Formulário de Triagem")
    pred_tab1, pred_tab2, pred_tab3 = st.tabs([
        "**Informações do Paciente**", 
        "**Histórico e Hábitos**", 
        "**Exames Laboratoriais**"
    ])

    with pred_tab1:
        st.header("Dados Demográficos e Condições Gerais")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_data['age'] = st.number_input('Idade', min_value=1, max_value=120, value=50, key='age_predict')
            patient_data['gender'] = st.selectbox('Gênero', ['Male', 'Female'], help="Selecione o gênero do paciente.", key='gender_predict')
            patient_data['region'] = st.selectbox('Região', ['Urban', 'Rural'], key='region_predict')
        with col2:
            patient_data['income_level'] = st.selectbox('Nível de Renda', ['Low', 'Middle', 'High'], key='income_predict')
            patient_data['hypertension'] = st.selectbox('Hipertensão', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='hyper_predict')
            patient_data['diabetes'] = st.selectbox('Diabetes', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='diab_predict')
        with col3:
            patient_data['obesity'] = st.selectbox('Obesidade', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='obesity_predict')
            patient_data['waist_circumference'] = st.number_input('Circunferência da Cintura (cm)', min_value=50, max_value=200, value=90, key='waist_predict')

    with pred_tab2:
        st.header("Estilo de Vida e Histórico Médico")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_data['family_history'] = st.selectbox('Histórico Familiar de Doença Cardíaca', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='family_predict')
            patient_data['smoking_status'] = st.selectbox('Status de Fumante', ['Never', 'Past', 'Current'], key='smoking_predict')
            patient_data['alcohol_consumption'] = st.selectbox('Consumo de Álcool', ['Never', 'Moderate', 'High'], key='alcohol_predict')
        with col2:
            patient_data['physical_activity'] = st.selectbox('Nível de Atividade Física', ['Low', 'Moderate', 'High'], key='phys_predict')
            patient_data['dietary_habits'] = st.selectbox('Hábitos Alimentares', ['Healthy', 'Unhealthy'], key='diet_predict')
            patient_data['stress_level'] = st.selectbox('Nível de Estresse Percebido', ['Low', 'Moderate', 'High'], key='stress_predict')
        with col3:
            patient_data['sleep_hours'] = st.slider('Média de Horas de Sono', min_value=1.0, max_value=12.0, value=7.0, step=0.5, key='sleep_predict')
            patient_data['air_pollution_exposure'] = st.selectbox('Exposição à Poluição do Ar', ['Low', 'Moderate', 'High'], key='air_predict')
            patient_data['medication_usage'] = st.selectbox('Uso de Medicação para Coração', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='med_predict')

    with pred_tab3:
        st.header("Resultados de Exames")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Pressão Arterial")
            patient_data['blood_pressure_systolic'] = st.number_input('Sistólica (mmHg)', min_value=80, max_value=250, value=120, key='bps_predict')
            patient_data['blood_pressure_diastolic'] = st.number_input('Diastólica (mmHg)', min_value=40, max_value=150, value=80, key='bpd_predict')
            st.subheader("Glicemia")
            patient_data['fasting_blood_sugar'] = st.number_input('Açúcar no Sangue em Jejum (mg/dL)', min_value=50, max_value=300, value=100, key='fbs_predict')
        with col2:
            st.subheader("Colesterol (mg/dL)")
            patient_data['cholesterol_level'] = st.number_input('Nível Total', min_value=50, max_value=500, value=200, key='chol_predict')
            patient_data['cholesterol_hdl'] = st.number_input('HDL', min_value=10, max_value=150, value=50, key='hdl_predict')
            patient_data['cholesterol_ldl'] = st.number_input('LDL', min_value=30, max_value=300, value=130, key='ldl_predict')
            patient_data['triglycerides'] = st.number_input('Triglicerídeos', min_value=30, max_value=500, value=150, key='trig_predict')
        with col3:
            st.subheader("Outros Exames")
            patient_data['EKG_results'] = st.selectbox('Resultados de EKG', ['Normal', 'Abnormal'], key='ekg_predict')
            patient_data['previous_heart_disease'] = st.selectbox('Doença Cardíaca Prévia Diagnosticada', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='prev_predict')
            patient_data['participated_in_free_screening'] = st.selectbox('Participou de Triagem Gratuita', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='screen_predict')

    # --- BOTÃO E RESULTADO DA PREDIÇÃO ---
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 3, 2])

    with col2:
        if st.button('**🩺 Analisar Risco Cardíaco**', use_container_width=True):
            if None in (model, scaler, label_encoders, feature_names):
                st.error("⚠️ Artefatos não carregados. Treine o modelo antes de realizar predições.")
            else:
                with st.spinner('Avaliando dados e calculando o risco...'):
                    resultado = predict_heart_attack_risk(patient_data, model, scaler, label_encoders, feature_names)
                st.subheader('Resultado da Análise de Risco')

                if resultado['prediction'] == 1:
                    st.error(f"**Classificação: {resultado['risk_label']}**", icon="💔")
                else:
                    st.success(f"**Classificação: {resultado['risk_label']}**", icon="❤️")

                prob_df = pd.DataFrame({
                    'Categoria': ['Risco de Ataque Cardíaco', 'Sem Risco Iminente'],
                    'Probabilidade (%)': [resultado['probability_attack'], resultado['probability_no_attack']]
                })
                st.bar_chart(prob_df.set_index('Categoria'))

                st.info(f"**Probabilidade de TER um ataque cardíaco:** {resultado['probability_attack']:.2f}%")
                st.info(f"**Probabilidade de NÃO ter um ataque cardíaco:** {resultado['probability_no_attack']:.2f}%")

with main_tabs[1]:
    st.header("Formulário de Cadastro de Paciente")
    
    @st.cache_data
    def get_csv_columns():
        return pd.read_csv('dataset/heart_attack_prediction_indonesia.csv', nrows=0).columns.tolist()

    csv_columns = get_csv_columns()

    reg_tab1, reg_tab2, reg_tab3 = st.tabs([
        "**Informações do Paciente**", 
        "**Histórico e Hábitos**", 
        "**Exames Laboratoriais e Desfecho**"
    ])

    with reg_tab1:
        st.header("Dados Demográficos e Condições Gerais")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_patient_data['age'] = st.number_input('Idade', min_value=1, max_value=120, value=50, key='age_reg')
            new_patient_data['gender'] = st.selectbox('Gênero', ['Male', 'Female'], help="Selecione o gênero do paciente.", key='gender_reg')
            new_patient_data['region'] = st.selectbox('Região', ['Urban', 'Rural'], key='region_reg')
        with col2:
            new_patient_data['income_level'] = st.selectbox('Nível de Renda', ['Low', 'Middle', 'High'], key='income_reg')
            new_patient_data['hypertension'] = st.selectbox('Hipertensão', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='hyper_reg')
            new_patient_data['diabetes'] = st.selectbox('Diabetes', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='diab_reg')
        with col3:
            new_patient_data['obesity'] = st.selectbox('Obesidade', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='obesity_reg')
            new_patient_data['waist_circumference'] = st.number_input('Circunferência da Cintura (cm)', min_value=50, max_value=200, value=90, key='waist_reg')

    with reg_tab2:
        st.header("Estilo de Vida e Histórico Médico")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_patient_data['family_history'] = st.selectbox('Histórico Familiar de Doença Cardíaca', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='family_reg')
            new_patient_data['smoking_status'] = st.selectbox('Status de Fumante', ['Never', 'Past', 'Current'], key='smoking_reg')
            new_patient_data['alcohol_consumption'] = st.selectbox('Consumo de Álcool', ['Never', 'Moderate', 'High'], key='alcohol_reg')
        with col2:
            new_patient_data['physical_activity'] = st.selectbox('Nível de Atividade Física', ['Low', 'Moderate', 'High'], key='phys_reg')
            new_patient_data['dietary_habits'] = st.selectbox('Hábitos Alimentares', ['Healthy', 'Unhealthy'], key='diet_reg')
            new_patient_data['stress_level'] = st.selectbox('Nível de Estresse Percebido', ['Low', 'Moderate', 'High'], key='stress_reg')
        with col3:
            new_patient_data['sleep_hours'] = st.slider('Média de Horas de Sono', min_value=1.0, max_value=12.0, value=7.0, step=0.5, key='sleep_reg')
            new_patient_data['air_pollution_exposure'] = st.selectbox('Exposição à Poluição do Ar', ['Low', 'Moderate', 'High'], key='air_reg')
            new_patient_data['medication_usage'] = st.selectbox('Uso de Medicação para Coração', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='med_reg')

    with reg_tab3:
        st.header("Resultados de Exames e Desfecho Clínico")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Pressão Arterial")
            new_patient_data['blood_pressure_systolic'] = st.number_input('Sistólica (mmHg)', min_value=80, max_value=250, value=120, key='bps_reg')
            new_patient_data['blood_pressure_diastolic'] = st.number_input('Diastólica (mmHg)', min_value=40, max_value=150, value=80, key='bpd_reg')
            st.subheader("Glicemia")
            new_patient_data['fasting_blood_sugar'] = st.number_input('Açúcar no Sangue em Jejum (mg/dL)', min_value=50, max_value=300, value=100, key='fbs_reg')
        with col2:
            st.subheader("Colesterol (mg/dL)")
            new_patient_data['cholesterol_level'] = st.number_input('Nível Total', min_value=50, max_value=500, value=200, key='chol_reg')
            new_patient_data['cholesterol_hdl'] = st.number_input('HDL', min_value=10, max_value=150, value=50, key='hdl_reg')
            new_patient_data['cholesterol_ldl'] = st.number_input('LDL', min_value=30, max_value=300, value=130, key='ldl_reg')
            new_patient_data['triglycerides'] = st.number_input('Triglicerídeos', min_value=30, max_value=500, value=150, key='trig_reg')
        with col3:
            st.subheader("Outros Exames e Desfecho")
            new_patient_data['EKG_results'] = st.selectbox('Resultados de EKG', ['Normal', 'Abnormal'], key='ekg_reg')
            new_patient_data['previous_heart_disease'] = st.selectbox('Doença Cardíaca Prévia Diagnosticada', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='prev_reg')
            new_patient_data['participated_in_free_screening'] = st.selectbox('Participou de Triagem Gratuita', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='screen_reg')
            new_patient_data['heart_attack'] = st.selectbox('Ocorrência de Ataque Cardíaco (Desfecho)', [0, 1], format_func=lambda x: 'Sim' if x == 1 else 'Não', key='attack_reg')

    # --- BOTÃO PARA SALVAR NOVOS DADOS ---
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button('**💾 Salvar Dados do Novo Paciente**', use_container_width=True):
            try:
                new_patient_df = pd.DataFrame([new_patient_data])
                new_patient_df = new_patient_df[csv_columns]
                new_patient_df.to_csv('dataset/heart_attack_prediction_indonesia.csv', mode='a', header=False, index=False)
                st.success("✅ Dados do novo paciente salvos com sucesso!")
                st.balloons()
            except Exception as e:
                st.error(f"❌ Ocorreu um erro ao salvar os dados: {e}")

with main_tabs[2]:
    st.header("Treinamento do Modelo")
    st.info("Clique no botão abaixo para iniciar um novo treinamento do modelo com os dados mais recentes do dataset.")
    
    col1, col2, col3 = st.columns([2, 3, 2])
    with col2:
        if st.button("**🚀 Iniciar Treinamento do Modelo**", use_container_width=True):
            try:
                train_model()
                st.success("✅ Modelo treinado e artefatos atualizados com sucesso!")
                st.warning("A aplicação será recarregada para usar o novo modelo.")
                # Limpa o cache para forçar o reload dos artefatos
                st.cache_resource.clear()
                # Força o recarregamento da página
                st.rerun()
            except Exception as e:
                st.error(f"❌ Ocorreu um erro durante o treinamento: {e}")


# --- RODAPÉ ---
st.markdown("---")
st.warning(
    "**Aviso Legal:** Esta é uma ferramenta de triagem baseada em um modelo de Machine Learning "
    "e não substitui, de forma alguma, a consulta, o diagnóstico ou o tratamento médico profissional. "
    "Consulte sempre um médico qualificado para questões de saúde."
)
