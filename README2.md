# Guia de Apresentação: Predição de Ataque Cardíaco com Machine Learning

## Visão Geral do Projeto
Sistema de suporte ao diagnóstico médico utilizando Machine Learning para prever probabilidade de ataque cardíaco baseado em dados clínicos e demográficos da população indonésia.

**Dataset:** Heart Attack Prediction Indonesia (258.355 registros, 28 features)
**Modelo Selecionado:** Random Forest (F1-Score: 0.4846, ROC-AUC: 0.6826)

---

## Roteiro de Apresentação

### 1. CONTEXTO E PROBLEMA DE NEGÓCIO
**Duração estimada: 3-5 min**

**Pontos a discutir:**
- Doenças cardiovasculares são a principal causa de morte no mundo
- Importância da detecção precoce e triagem de pacientes de alto risco
- Objetivo: desenvolver ferramenta de apoio à decisão clínica (não substituição)
- Uso prático: triagem inicial, priorização de casos urgentes

**Questões para discussão:**
- Como este modelo pode ser integrado no fluxo de trabalho hospitalar?
- Quais são os riscos éticos de usar ML em diagnóstico médico?

---

### 2. EXPLORAÇÃO E ANÁLISE DOS DADOS
**Duração estimada: 8-10 min**

#### 2.1 Características do Dataset
- **Tamanho:** 258.355 pacientes, 28 variáveis (17 numéricas, 10 categóricas)
- **Memória:** 166.96 MB
- **Target:** 59.81% sem ataque, 40.19% com ataque (relativamente balanceado)

**Pontos a destacar:**
- Ausência de duplicatas (qualidade dos dados)
- Apenas 1 variável com missing values: alcohol_consumption (59.95% ausente)
- Problema leve de dados: cholesterol_ldl com 22 valores negativos

#### 2.2 Variáveis Categóricas (10 features)
- Demográficas: gender, region, income_level
- Estilo de vida: smoking_status, alcohol_consumption, physical_activity, dietary_habits
- Ambiente/Saúde: air_pollution_exposure, stress_level, EKG_results

**Insight importante:**
- Variáveis de estilo de vida mostram correlação visível com ataque cardíaco
- Hábitos saudáveis, atividade física alta = menor taxa de ataque

#### 2.3 Variáveis Numéricas (17 features)
**Principais:**
- Idade, circunferência abdominal, pressão arterial (sistólica/diastólica)
- Níveis de colesterol (total, HDL, LDL), triglicerídeos
- Glicemia em jejum, horas de sono

**Discussão:**
- Distribuições mostram sobreposição significativa entre pacientes com/sem ataque
- Boxplots revelam outliers em várias variáveis (normal em dados médicos)
- Nenhuma variável sozinha discrimina perfeitamente o target

#### 2.4 Análise de Correlação
**Top 5 variáveis mais correlacionadas com ataque cardíaco:**
1. obesity (correlação positiva)
2. cholesterol_ldl (correlação positiva)
3. previous_heart_disease (correlação positiva)
4. hypertension (correlação positiva)
5. diabetes (correlação positiva)

**Ponto de discussão:**
- Correlações são relativamente baixas (~10-15%)
- Indica que o fenômeno é multifatorial (nenhum fator sozinho determina)
- Justifica uso de ML para capturar interações complexas

---

### 3. PRÉ-PROCESSAMENTO E PREPARAÇÃO DOS DADOS
**Duração estimada: 4-5 min**

#### Etapas realizadas:
1. **Verificação de duplicatas:** nenhuma encontrada
2. **Tratamento de inconsistências:** identificados valores negativos em cholesterol_ldl
3. **Encoding de categóricas:** LabelEncoder para 10 variáveis
4. **Divisão dos dados:**
   - Treino: 70% (180.848 amostras)
   - Validação: 15% (38.753 amostras)
   - Teste: 15% (38.754 amostras)
5. **Normalização:** StandardScaler (média=0, std=1)

**Pontos importantes:**
- Uso de stratify para manter distribuição do target em todos os conjuntos
- Separação treino/validação/teste permite avaliar generalização
- Normalização essencial para modelos sensíveis à escala (KNN, SVM, Logistic Regression)

---

### 4. MODELAGEM E COMPARAÇÃO DE ALGORITMOS
**Duração estimada: 6-8 min**

#### 4.1 Modelos Testados
1. **Logistic Regression** - baseline linear
2. **Decision Tree** - modelo não-linear simples
3. **Random Forest** - ensemble de árvores
4. **KNN** - baseado em proximidade

#### 4.2 Resultados (conjunto de validação)

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| Random Forest | 0.652 | 0.599 | 0.407 | **0.485** | 0.682 |
| Logistic Regression | 0.655 | 0.612 | 0.387 | 0.474 | **0.685** |
| Decision Tree | 0.566 | 0.461 | 0.476 | 0.468 | 0.551 |
| KNN | 0.603 | 0.508 | 0.383 | 0.437 | 0.600 |

**Discussão crítica:**
- Random Forest venceu por F1-Score (balanço entre precision e recall)
- Logistic Regression teve melhor ROC-AUC (discriminação geral)
- Todos os modelos apresentam recall baixo (~38-47%) - problema crítico!
- Trade-off: aumentar recall pode diminuir precision (mais alarmes falsos)

#### 4.3 Escolha da Métrica
**Por que F1-Score?**
- Balanceia Precision e Recall
- Importante em contexto médico onde ambos os erros têm custo

**Discussão:**
- Em triagem médica, Recall pode ser mais importante que Precision
- Falso Negativo = paciente de risco não detectado (potencialmente fatal)
- Falso Positivo = exames desnecessários (custo financeiro, ansiedade)
- Modelo atual prioriza Precision, sacrificando Recall

**Pergunta provocativa:**
- Seria melhor ter um modelo com Recall 80% e Precision 40%?

---

### 5. AVALIAÇÃO DO MODELO FINAL (RANDOM FOREST)
**Duração estimada: 7-10 min**

#### 5.1 Métricas no Conjunto de Teste
- **Accuracy:** 65.38% - modelo acerta ~2/3 dos casos
- **Precision:** 60.18% - quando prediz ataque, acerta 60% das vezes
- **Recall:** 40.96% - detecta apenas 41% dos ataques reais
- **F1-Score:** 48.74%
- **ROC-AUC:** 68.26% - capacidade moderada de discriminação

#### 5.2 Matriz de Confusão (análise detalhada)
**Resultados absolutos:**
- Verdadeiros Negativos (TN): 19.049 - sem ataque, corretamente identificados
- Falsos Positivos (FP): 4.129 - sem ataque, identificados como com ataque (17.81%)
- Falsos Negativos (FN): 9.196 - **COM ATAQUE, NÃO DETECTADOS (59.04%)** ⚠️
- Verdadeiros Positivos (TP): 6.380 - com ataque, corretamente identificados

**PONTO CRÍTICO DE DISCUSSÃO:**
- **59% dos pacientes com risco de ataque NÃO são detectados!**
- Este é o maior problema do modelo atual
- Para uso clínico, isso seria inaceitável

**Possíveis soluções:**
1. Ajustar threshold de decisão (reduzir de 0.5 para 0.3, por exemplo)
2. Treinar modelo com class_weight balanceado
3. Usar técnicas de oversampling/undersampling
4. Aceitar mais falsos positivos para capturar mais verdadeiros positivos

#### 5.3 Curva ROC
- AUC = 0.6826 (moderado)
- Permite visualizar trade-off entre TPR e FPR em diferentes thresholds
- Gráfico mostra que há espaço para melhorar escolhendo threshold ótimo

---

### 6. INTERPRETABILIDADE DO MODELO
**Duração estimada: 5-7 min**

#### 6.1 Feature Importance (Top 10)
As variáveis mais importantes para o Random Forest são:

1. **age** - idade do paciente
2. **blood_pressure_systolic** - pressão sistólica
3. **cholesterol_level** - nível total de colesterol
4. **sleep_hours** - horas de sono
5. **waist_circumference** - circunferência abdominal
6. **blood_pressure_diastolic** - pressão diastólica
7. **fasting_blood_sugar** - glicemia em jejum
8. **cholesterol_ldl** - colesterol ruim
9. **triglycerides** - triglicerídeos
10. **cholesterol_hdl** - colesterol bom

**Observações:**
- Idade é o fator mais importante (esperado clinicamente)
- Métricas cardiovasculares (pressão, colesterol) são muito relevantes
- Horas de sono aparecem no top 5 (insight interessante!)
- Fatores de estilo de vida têm importância menor no modelo

**Discussão:**
- Resultados fazem sentido clínico?
- Surpreendente que diabetes e obesity não apareçam no top 10
- Possível multicolinearidade entre features

---

### 7. FUNÇÃO DE PREDIÇÃO PARA NOVOS PACIENTES
**Duração estimada: 3-4 min**

**Funcionalidade:**
- Sistema permite entrada de dados de novo paciente
- Retorna probabilidade de ataque cardíaco
- Classificação: ALTO RISCO vs BAIXO RISCO

**Exemplo de uso:**
- Paciente com perfil de alto risco foi testado
- Modelo pode fornecer probabilidades para auxiliar decisão médica

**Importante frisar:**
- Ferramenta de APOIO, não de decisão final
- Médico deve sempre ter palavra final
- Complementa avaliação clínica tradicional

---

### 8. DISCUSSÃO CRÍTICA E LIMITAÇÕES
**Duração estimada: 8-10 min**

#### 8.1 Pontos Fortes do Modelo
✓ Utiliza dados reais de larga escala (258k pacientes)
✓ Abordagem sistemática e reproduzível
✓ Interpretabilidade através de feature importance
✓ Pode auxiliar em triagem inicial
✓ Potencial de integração em sistemas hospitalares

#### 8.2 Limitações Críticas
⚠️ **Recall muito baixo (41%)** - não detecta maioria dos casos de ataque
⚠️ **Viés geográfico** - dados da Indonésia podem não generalizar
⚠️ **Missing data** - 60% sem informação de consumo de álcool
⚠️ **Fatores não capturados** - genética, histórico familiar detalhado
⚠️ **Desbalanceamento não tratado** - poderia usar técnicas de balanceamento
⚠️ **Sem validação clínica prospectiva**

#### 8.3 Riscos e Considerações Éticas
1. **Risco de falsa segurança:** paciente de alto risco não detectado
2. **Viés algorítmico:** modelo pode ter performance diferente por etnia/região
3. **Responsabilidade legal:** quem é responsável por erro do modelo?
4. **Privacidade:** dados de saúde são sensíveis
5. **Equidade:** modelo pode perpetuar desigualdades em saúde

#### 8.4 Aplicabilidade Prática
**Cenários adequados:**
- Triagem de grandes populações (programas de saúde pública)
- Priorização de pacientes em filas de atendimento
- Alertas automáticos em prontuários eletrônicos
- Ferramentas de educação médica

**Cenários inadequados:**
- Diagnóstico definitivo sem avaliação médica
- Decisões de tratamento baseadas apenas no modelo
- Uso em populações muito diferentes da amostra de treino

---

### 9. PRÓXIMOS PASSOS E MELHORIAS
**Duração estimada: 4-5 min**

#### Melhorias Técnicas
1. **Ajuste de threshold** para aumentar Recall
2. **Balanceamento de classes** (SMOTE, class_weight)
3. **Hyperparameter tuning** (GridSearchCV, RandomSearchCV)
4. **Feature engineering** - criar interações, features polinomiais
5. **Modelos mais avançados** - XGBoost, LightGBM, Neural Networks
6. **Ensemble** - combinar múltiplos modelos

#### Melhorias de Dados
1. Coletar dados de múltiplas regiões/países
2. Incluir mais features clínicas (ECG detalhado, biomarcadores)
3. Dados longitudinais (acompanhamento temporal)
4. Tratar missing values de forma mais sofisticada

#### Validação e Deployment
1. **Validação clínica prospectiva** em ambiente hospitalar
2. **Estudos de impacto** - o modelo realmente melhora desfechos?
3. **Interface amigável** para profissionais de saúde
4. **Integração com HIS/EMR** (sistemas hospitalares)
5. **Monitoramento contínuo** - performance degrada com o tempo?
6. **Feedback loop** - médicos corrigem predições, modelo aprende

---

## Perguntas para Discussão Final

1. **Ética:** É ético usar este modelo com Recall de apenas 41% em contexto clínico?

2. **Trade-offs:** Você preferiria um modelo com 80% Recall e 40% Precision, ou manter o atual?

3. **Confiança:** Em que ponto você confiaria neste modelo para tomar decisões sobre sua própria saúde?

4. **Regulamentação:** Que tipo de aprovação regulatória seria necessária para usar isto em hospitais?

5. **Responsabilidade:** Se o modelo falha em detectar um ataque cardíaco, quem é responsável?

6. **Generalização:** Como garantir que o modelo funciona bem em populações diferentes?

7. **Implementação:** Que barreiras práticas existem para implementar isto em hospitais brasileiros?

8. **Futuro:** Machine Learning vai substituir médicos? Por que sim/não?

---

## Conclusão

Este projeto demonstra:
- ✓ Aplicação prática de ML em saúde
- ✓ Metodologia completa de Data Science (EDA → Modelagem → Avaliação)
- ✓ Importância de métricas além de accuracy
- ✓ Necessidade de interpretabilidade em aplicações críticas
- ⚠️ Desafios éticos e práticos de ML em medicina

**Mensagem final:**
Machine Learning é uma ferramenta poderosa de apoio à decisão médica, mas deve ser usado com cautela, transparência e sempre sob supervisão humana qualificada.
