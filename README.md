# TECH CHALLENGE — FASE 1  
**Código:** #7IADTF1  

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
