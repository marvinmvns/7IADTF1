# 🩺 Sistema de Predição de Risco de Ataque Cardíaco

Sistema inteligente de triagem para avaliação de risco de ataque cardíaco em pacientes indonésios, utilizando Machine Learning com Random Forest.

# 🔥 Otimização na V2
Com o apoio do copilot meu modelo deu um salto consideravel na redução de falsos negativos, com implementação de:
- GridSearch com class_weight='balanced' 
- OneHotEncoder (representação correta de categóricas)
- Hiperparâmetros otimizados (n_estimators=200, max_depth=10)
- ColumnTransformer + Pipeline (prevenção de data leakage)
 Apesar disso, existe umn tradeoff na perda de precisão, para casos médicos ainda vale! Porém na apresentação do video foi feita sobre a V1.

## 📋 Sobre o Projeto

Este projeto faz parte do curso de **FIAP - 7IADT** e implementa um sistema completo de predição de risco cardíaco com:
- Interface web interativa para triagem de pacientes
- Modelo de Machine Learning (Random Forest) com interpretabilidade SHAP
- Sistema de cadastro de novos pacientes
- Pipeline de retreinamento do modelo
- Análise exploratória de dados completa

### 🎯 Objetivos
- Identificar pacientes com alto risco de ataque cardíaco
- Fornecer ferramenta de triagem acessível para profissionais de saúde
- Permitir análise e interpretação dos fatores de risco

## 📊 Dataset

**Fonte:** [Heart Attack Prediction in Indonesia - Kaggle](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia)

O dataset contém **28 features** incluindo:
- **Demográficos:** idade, gênero, região, nível de renda
- **Condições de saúde:** hipertensão, diabetes, obesidade, histórico familiar
- **Estilo de vida:** tabagismo, álcool, atividade física, dieta, sono
- **Exames laboratoriais:** pressão arterial, glicemia, colesterol, triglicerídeos, EKG

## 🚀 Tecnologias Utilizadas

- **Python 3.8+**
- **Streamlit** - Interface web interativa
- **scikit-learn** - Machine Learning (Random Forest)
- **pandas** - Manipulação de dados
- **numpy** - Operações numéricas
- **matplotlib & seaborn** - Visualização de dados
- **SHAP** - Interpretabilidade do modelo
- **Jupyter Notebook** - Análise exploratória

---

## 💻 Como Rodar no VS Code (Local)

### Pré-requisitos
- Python 3.8 ou superior instalado
- Git instalado
- VS Code (recomendado)

### Passo 1: Clonar o Repositório
```bash
git clone https://github.com/marvinmvns/7IADTF1.git
cd 7IADT
```

### Passo 2: Criar Ambiente Virtual (Recomendado)
```bash
# No Linux/Mac
python3 -m venv venv
source venv/bin/activate

# No Windows
python -m venv venv
venv\Scripts\activate
```

### Passo 3: Instalar Dependências
```bash
pip install -r requirements.txt
```

### Passo 4: Criar Pasta de Artefatos
```bash
mkdir -p artifacts
```

### Passo 5: Treinar o Modelo (Primeira Vez)
```bash
python3 train_model.py
```

Isso irá gerar os seguintes artefatos em `artifacts/`:
- `best_model.pkl` - Modelo treinado
- `scaler.pkl` - Scaler para normalização
- `label_encoders.pkl` - Encoders para variáveis categóricas
- `feature_names.pkl` - Nomes das features

### Passo 6: Executar a Aplicação Streamlit
```bash
streamlit run app.py
```

A aplicação abrirá automaticamente em `http://localhost:8501`

### 📝 Notas para VS Code
- Instale a extensão **Python** da Microsoft
- Instale a extensão **Jupyter** para trabalhar com notebooks
- Configure o interpretador Python para usar o ambiente virtual criado

---

## 🐳 Como Rodar com Docker

### Passo 1: Build da imagem
```bash
docker build -t triagem-cardio .
```

### Passo 2: (Opcional) Treinar o modelo dentro do contêiner
Monte a pasta `artifacts/` para persistir os modelos e execute o script de treino.
```bash
docker run --rm -v "$(pwd)/artifacts:/app/artifacts" triagem-cardio python train_model.py
```

### Passo 3: Subir a aplicação Streamlit
```bash
docker run --rm -p 8501:8501 -v "$(pwd)/artifacts:/app/artifacts" triagem-cardio
```

A aplicação ficará disponível em `http://localhost:8501`.

> 💡 No Windows PowerShell, substitua `$(pwd)` por `${PWD}`. Em shells baseados em CMD, utilize o caminho completo, como `C:\caminho\para\7IADT\artifacts`.

---

## ☁️ Como Rodar no Google Colab

### Opção 1: Apenas Análise Exploratória e Treinamento

1. **Acesse o Google Colab:** [https://colab.research.google.com/](https://colab.research.google.com/)

2. **Faça upload do notebook:**
   - Clique em "Arquivo" → "Fazer upload do notebook"
   - Selecione o arquivo `ataquecardiaco.ipynb`

3. **Faça upload do dataset:**
   - Crie uma pasta `dataset` no Colab
   - Faça upload do arquivo `heart_attack_prediction_indonesia.csv`

4. **Execute as células sequencialmente**

### Opção 2: Executar a Aplicação Streamlit no Colab

1. **Crie um novo notebook no Colab**

2. **Clone o repositório:**
```python
!git clonehttps://github.com/marvinmvns/7IADTF1.git
%cd 7IADT
```

3. **Instale as dependências:**
```python
!pip install -q streamlit pandas scikit-learn numpy matplotlib seaborn shap tqdm pyngrok
```

4. **Configure o ngrok para expor o Streamlit:**
```python
!pip install -q pyngrok

from pyngrok import ngrok

# Configure seu token do ngrok (obtenha em https://dashboard.ngrok.com/get-started/your-authtoken)
!ngrok authtoken SEU_TOKEN_AQUI
```

5. **Treine o modelo (se necessário):**
```python
!python3 train_model.py
```

6. **Execute o Streamlit com ngrok:**
```python
# Em uma célula separada
!streamlit run app.py &>/dev/null &

# Em outra célula
public_url = ngrok.connect(8501)
print(f"Acesse a aplicação em: {public_url}")
```

### 📌 Dicas para Google Colab
- O Colab tem limite de tempo de execução (pode desconectar após inatividade)
- Os arquivos são temporários - faça backup dos artefatos gerados
- Para usar GPU: "Ambiente de execução" → "Alterar tipo de ambiente" → GPU
- O ngrok é necessário pois o Colab não expõe portas diretamente

---

## 📁 Estrutura do Projeto

```
7IADT/
├── app.py                          # Aplicação Streamlit principal
├── train_model.py                  # Script de treinamento do modelo
├── ataquecardiaco.ipynb           # Notebook com análise exploratória e SHAP
├── requirements.txt                # Dependências do projeto
├── CLAUDE.md                       # Documentação para Claude Code
├── README.md                       # Este arquivo
├── TODO.md                         # Checklist do projeto
│
├── dataset/
│   └── heart_attack_prediction_indonesia.csv  # Dataset principal
│
└── artifacts/                      # Artefatos gerados (após treinamento)
    ├── best_model.pkl
    ├── scaler.pkl
    ├── label_encoders.pkl
    └── feature_names.pkl
```

---

## 🎮 Como Usar a Aplicação

### 1️⃣ Aba: Triagem de Risco
- Preencha os dados do paciente nos formulários organizados por categoria
- Clique em "Analisar Risco Cardíaco"
- Visualize a classificação de risco e probabilidades

### 2️⃣ Aba: Cadastrar Novo Paciente
- Preencha todos os dados do paciente incluindo o desfecho (se teve ataque cardíaco)
- Clique em "Salvar Dados do Novo Paciente"
- Os dados serão adicionados ao dataset principal

### 3️⃣ Aba: Treinar Modelo
- Clique em "Iniciar Treinamento do Modelo"
- Aguarde o processo de treinamento
- O modelo será retreinado com todos os dados disponíveis
- A aplicação recarregará automaticamente com o novo modelo

---

## 📊 Funcionalidades

✅ **Predição de Risco em Tempo Real**
- Classificação binária: Alto Risco / Baixo Risco
- Probabilidades percentuais para cada classe
- Visualização gráfica dos resultados

✅ **Gerenciamento de Dados**
- Cadastro de novos pacientes
- Dados salvos diretamente no CSV
- Preservação da estrutura do dataset

✅ **Pipeline ML Completo**
- Pré-processamento automático
- Encoding de variáveis categóricas
- Normalização de features
- Modelo Random Forest otimizado

✅ **Retreinamento do Modelo**
- Treinamento via interface ou linha de comando
- Atualização automática dos artefatos
- Logs de progresso em tempo real

✅ **Interpretabilidade (Notebook)**
- Análise SHAP para explicabilidade
- Importância de features
- Visualizações detalhadas

---

## 🔬 Metodologia

### Pré-processamento
1. Codificação de variáveis categóricas (LabelEncoder)
2. Divisão treino-teste (70-30) com estratificação
3. Normalização com StandardScaler

### Modelagem
- **Algoritmo:** Random Forest Classifier
- **Parâmetros:** 100 estimadores, random_state=42
- **Métricas:** Acurácia, Precisão, Recall, F1-Score, AUC-ROC

### Interpretabilidade
- **SHAP (SHapley Additive exPlanations)** para entender as contribuições de cada feature

---

## ⚠️ Aviso Legal

**Esta é uma ferramenta de triagem baseada em Machine Learning e NÃO substitui consulta, diagnóstico ou tratamento médico profissional.**

Sempre consulte um médico qualificado para questões de saúde. Esta ferramenta foi desenvolvida para fins educacionais e de pesquisa.

---

## 🎓 Referências

- [Kaggle Dataset](https://www.kaggle.com/datasets/ankushpanday2/heart-attack-prediction-in-indonesia)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [SHAP Documentation](https://shap.readthedocs.io/)

