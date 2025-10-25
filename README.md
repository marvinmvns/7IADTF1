# TECH CHALLENGE — FASE 1  


---

## Descrição Geral

O **Tech Challenge** é um projeto integrador que reúne os conhecimentos adquiridos em todas as disciplinas da fase.  
A atividade é **obrigatória** e corresponde a **90% da nota** das disciplinas.  
Deve ser desenvolvida **em grupo** e entregue dentro do **prazo estipulado**.

---

## Desafio

Um **hospital universitário** busca implementar um **sistema inteligente de suporte ao diagnóstico**, com o objetivo de:  
- Ajudar médicos e equipes clínicas na **análise inicial de exames**;  
- Apoiar decisões médicas por meio de **processamento de dados médicos**;  
- **Reduzir erros** e **otimizar o tempo** dos profissionais da saúde.  

Nesta primeira fase, o desafio é **criar a base do sistema de IA** utilizando **Machine Learning (ML)** para analisar resultados de exames automaticamente e destacar informações relevantes para o diagnóstico.

---

## Objetivo

Construir uma **solução inicial de IA** voltada ao **processamento de exames médicos e documentos clínicos**, aplicando fundamentos essenciais de:
- **Inteligência Artificial (IA)**  
- **Machine Learning (ML)**  
- **Visão Computacional**

---

## Entregas Técnicas

### 1. Processamento de Dados Médicos

#### Classificação de Exames com Machine Learning
- Escolher uma **base de dados tabular**;  
- Criar um **modelo de diagnóstico binário** (“a pessoa tem ou não uma doença”) utilizando algoritmos de aprendizado de máquina.

#### Extra (Opcional)
- Implementar diagnóstico com **dados de imagem** (ex: radiografias, tomografias);  
- Utilizar **redes neurais convolucionais (CNNs)**;  
- Esta etapa **não é obrigatória**, mas pode **aumentar a nota** final.

---

### 2. Dados e Modelos

#### Escolha de Datasets
- Selecionar um ou mais **datasets médicos públicos**;  
- **Discutir o problema** a ser resolvido.

#### Exploração de Dados
- Carregar a base de dados e explorar suas características;  
- Analisar **estatísticas descritivas** e **distribuições**;  
- Discutir os **resultados obtidos**.

#### Pré-processamento de Dados
- Realizar **limpeza dos dados**, tratando valores ausentes e inconsistentes;  
- Construir um **pipeline de pré-processamento** em **Python**:
  - Converter variáveis **categóricas e numéricas** em formatos adequados;
  - Realizar **análise de correlação** entre as variáveis.

---

### 3. Modelagem

#### Criação dos Modelos
- Desenvolver **modelos preditivos de classificação** com **duas ou mais técnicas**, como:
  - Regressão Logística  
  - Árvore de Decisão  
  - KNN  
  - (ou outras de sua escolha)  
- Garantir **divisão clara** entre **treino**, **validação** e **teste**.

---

### 4. Treinamento e Avaliação

#### Treinamento
- Treinar o modelo com o **conjunto de treinamento**.

#### Avaliação
- Avaliar o modelo com o **conjunto de teste**, utilizando métricas como:
  - **Accuracy**
  - **Recall**
  - **F1-Score**

- Discutir a **escolha da métrica** considerando o contexto clínico;  
- Interpretar os resultados utilizando técnicas como:
  - **Feature Importance**
  - **SHAP Values**

#### Reflexão
- Discutir criticamente os resultados:  
  - O modelo pode ser aplicado na prática?  
  - Como ele auxiliaria o médico?  
  - Lembrar que **a decisão final é sempre médica**.

---

## Exemplos de Fontes de Dados

### Tarefas Principais
- **Diagnóstico de Câncer de Mama (maligno ou benigno):**  
  [Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data)
- **Diagnóstico de Diabetes:**  
  [Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set/data)
- Ou outro **dataset médico público** de sua escolha.

### Tarefas Extras — Visão Computacional
- **Detecção de Pneumonia em Radiografias:**  
  [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Detecção de Câncer de Mama (imagens):**  
  [CBIS-DDSM Dataset](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breastcancer-image-dataset/data)

---

## Código e Organização

- Projeto em **Python**, **estruturado e documentado**;  
- Utilizar **Jupyter Notebook** ou **scripts Python** para demonstração dos resultados.

---

## Entregáveis da Fase 1

### Arquivo PDF (com link para o repositório Git)
Deve conter:
- Código-fonte completo;  
- `Dockerfile` e `README.md` com instruções de execução;  
- Dataset (ou link para download);  
- Resultados obtidos (**prints, gráficos e análises**);  
- **Relatório técnico** descrevendo:
  - Estratégias de pré-processamento;  
  - Modelos utilizados e justificativa;  
  - Resultados e interpretação.

---

### Vídeo de Demonstração
- Upload no **YouTube** ou **Vimeo** (configuração **público** ou **não listado**);  
- **Duração máxima:** 15 minutos;  
- Deve apresentar:
  - Execução do sistema;  
  - Breve explicação do **fluxo de funcionamento**.

---

## Checklist Final

| Atividade | Status |
|------------|--------|
| Escolha do dataset médico | ☐ |
| Exploração e análise de dados | ☐ |
| Pré-processamento e pipeline | ☐ |
| Modelagem com ≥ 2 técnicas | ☐ |
| Treinamento e avaliação | ☐ |
| Interpretação e discussão | ☐ |
| Entrega no GitHub (PDF, código, Docker, README) | ☐ |
| Vídeo demonstrativo (YouTube/Vimeo) | ☐ |



# Projeto de Previsão de Risco de Ataque Cardíaco

Este projeto é uma aplicação web interativa construída com Streamlit para prever o risco de ataque cardíaco com base em dados do paciente. Ele utiliza um modelo de Machine Learning (Random Forest) treinado em um conjunto de dados de saúde da Indonésia.

## Como Executar o Projeto

Existem duas maneiras principais de executar este projeto: localmente usando o Visual Studio Code ou na nuvem usando o Google Colab.

---

### 1. Executando Localmente com VS Code

Esta abordagem é recomendada para desenvolvimento e para ter controle total sobre o ambiente.

#### Pré-requisitos

*   **Python 3.8+:** Certifique-se de ter o Python instalado. Você pode baixá-lo em [python.org](https://www.python.org/downloads/).
*   **Visual Studio Code:** Um editor de código-fonte gratuito e poderoso. Baixe em [code.visualstudio.com](https://code.visualstudio.com/).
*   **Git:** Para clonar o repositório. Baixe em [git-scm.com](https://git-scm.com/downloads).

#### Passos para Configuração

1.  **Clonar o Repositório:**
    Abra um terminal e clone o repositório para a sua máquina local.

    ```bash
    git clone <URL_DO_REPOSITORIO>
    cd 7IADTF1
    ```

2.  **Abrir no VS Code:**
    Abra a pasta do projeto no VS Code.

    ```bash
    code .
    ```

3.  **Criar e Ativar um Ambiente Virtual:**
    É uma boa prática usar um ambiente virtual para isolar as dependências do projeto. Abra o terminal integrado do VS Code (`Ctrl+` ou `View > Terminal`).

    ```bash
    # Criar o ambiente virtual
    python -m venv .venv

    # Ativar o ambiente virtual
    # No Windows:
    .venv\Scripts\activate
    # No macOS/Linux:
    source .venv/bin/activate
    ```

4.  **Instalar as Dependências:**
    Com o ambiente virtual ativado, instale as bibliotecas Python necessárias a partir do arquivo `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

5.  **Executar a Aplicação Streamlit:**
    Após a instalação, inicie a aplicação.

    ```bash
    streamlit run app.py
    ```

    O VS Code pode perguntar se você deseja iniciar a aplicação. A aplicação será aberta automaticamente no seu navegador padrão.

---

### 2. Executando no Google Colab

Esta abordagem é ideal para quem não deseja instalar nada localmente e quer executar o projeto em um ambiente de notebook baseado em nuvem.

#### Pré-requisitos

*   Uma conta Google.

#### Passos para Configuração

1.  **Fazer o Upload dos Arquivos para o Google Drive:**
    *   Crie uma pasta no seu Google Drive (ex: `7IADTF1`).
    *   Faça o upload de todos os arquivos e pastas do projeto para esta nova pasta no Drive, mantendo a estrutura de diretórios original (`app.py`, `train_model.py`, `dataset/`, etc.).

2.  **Criar um Novo Notebook no Google Colab:**
    *   Acesse [colab.research.google.com](https://colab.research.google.com).
    *   Clique em `File > New notebook`.

3.  **Montar o Google Drive:**
    Execute a célula a seguir no notebook para dar ao Colab acesso aos seus arquivos no Google Drive.

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

4.  **Navegar para a Pasta do Projeto:**
    Use o comando `cd` para navegar até a pasta onde você fez o upload dos arquivos.

    ```python
    # Altere o caminho se você usou um nome de pasta diferente
    %cd /content/drive/MyDrive/7IADTF1
    ```

5.  **Instalar as Dependências:**
    Instale as bibliotecas necessárias no ambiente do Colab a partir do arquivo `requirements.txt`.

    ```python
    !pip install -r requirements.txt
    ```

6.  **Executar a Aplicação com `pyngrok`:**
    O Google Colab não expõe portas diretamente. Usaremos o `pyngrok` para criar um túnel público para a nossa aplicação Streamlit.

    *   **Treinar o modelo (se necessário):**
        Se os artefatos do modelo (`.pkl`) não estiverem presentes, treine o modelo primeiro.

        ```python
        !python train_model.py
        ```

    *   **Executar a aplicação:**
        Crie um arquivo chamado `run_streamlit.py` (ou adicione este código a uma célula do notebook) com o seguinte conteúdo:

        ```python
        from pyngrok import ngrok
        import subprocess

        # Inicia o túnel ngrok na porta 8501 (padrão do Streamlit)
        public_url = ngrok.connect(8501)
        print(f"URL pública do Streamlit: {public_url}")

        # Inicia a aplicação Streamlit em segundo plano
        process = subprocess.Popen(['streamlit', 'run', 'app.py'])
        process.wait()
        ```

    Execute este script/célula. A saída fornecerá uma URL pública (geralmente terminando com `.ngrok.io`). Clique nessa URL para interagir com sua aplicação Streamlit diretamente do seu navegador.

## Funcionalidades da Aplicação

*   **Triagem de Risco:** Preencha um formulário com os dados do paciente para obter uma previsão de risco de ataque cardíaco.
*   **Cadastrar Novo Paciente:** Adicione novos dados de pacientes ao dataset, que podem ser usados para retreinar o modelo.
*   **Treinar Modelo:** Inicie um novo processo de treinamento do modelo com os dados mais recentes.# 7IADTF1
