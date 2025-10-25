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


## 12. Roadmap de Melhorias e Próximos Passos

Esta seção apresenta um plano estruturado para evolução do projeto, organizado por prioridade e complexidade.

### 12.1 Melhorias de Curto Prazo (1-2 semanas)

**1. Otimização de Hiperparâmetros**

Atualmente usamos os hiperparâmetros padrão. Podemos melhorar significativamente com Grid Search ou Random Search:

```python
from sklearn.model_selection import GridSearchCV

# Definir grid de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Grid Search com cross-validation
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_scaled, y_train)

print(f"Melhores hiperparâmetros: {grid_search.best_params_}")
print(f"Melhor F1-Score: {grid_search.best_score_:.4f}")
```

**Alternativa mais eficiente - Random Search:**

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(100, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,  # 50 combinações aleatórias
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
```

**2. Feature Engineering Avançado**

Implementar as features derivadas mencionadas na seção 11.2:

```python
def create_advanced_features(df):
    """Criar features derivadas clinicamente relevantes"""
    
    df_enhanced = df.copy()
    
    # Índices clínicos
    df_enhanced['cholesterol_ratio'] = df['cholesterol_level'] / (df['cholesterol_hdl'] + 1e-10)
    df_enhanced['cholesterol_ldl_hdl_ratio'] = df['cholesterol_ldl'] / (df['cholesterol_hdl'] + 1e-10)
    
    # Pressão de pulso (indicador de rigidez arterial)
    df_enhanced['pulse_pressure'] = df['blood_pressure_systolic'] - df['blood_pressure_diastolic']
    
    # Score de risco combinado
    df_enhanced['risk_score'] = (
        df['age'] / 100 * 0.2 +
        df['cholesterol_level'] / 300 * 0.2 +
        df['blood_pressure_systolic'] / 180 * 0.2 +
        df['obesity'] * 0.2 +
        df['diabetes'] * 0.1 +
        df['hypertension'] * 0.1
    )
    
    # Interações
    df_enhanced['age_cholesterol'] = df['age'] * df['cholesterol_level'] / 10000
    df_enhanced['comorbidity_count'] = df['diabetes'] + df['hypertension'] + df['obesity']
    
    # Categorização de idade
    df_enhanced['age_risk_category'] = pd.cut(
        df['age'],
        bins=[0, 40, 55, 70, 120],
        labels=[0, 1, 2, 3]  # Baixo, Moderado, Alto, Muito Alto
    ).astype(int)
    
    return df_enhanced

# Aplicar
df = create_advanced_features(df)
```

**3. Análise de Feature Importance Refinada**

```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Permutation Importance (mais confiável que feature_importances_)
perm_importance = permutation_importance(
    best_model, 
    X_test_scaled, 
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Visualização
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(range(20), feature_importance_df['importance_mean'][:20])
plt.yticks(range(20), feature_importance_df['feature'][:20])
plt.xlabel('Permutation Importance')
plt.title('Top 20 Features - Permutation Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
```

### 12.2 Melhorias de Médio Prazo (1-2 meses)

**1. Implementar Modelos Avançados**

```python
# XGBoost
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    use_label_encoder=False
)

xgb_model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],
              early_stopping_rounds=10,
              verbose=False)

# LightGBM (mais rápido que XGBoost)
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary',
    random_state=42
)

lgb_model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],
              callbacks=[lgb.early_stopping(10)])

# CatBoost (lida bem com categóricas)
from catboost import CatBoostClassifier

cat_model = CatBoostClassifier(
    iterations=200,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    random_state=42,
    verbose=False
)

cat_model.fit(X_train_scaled, y_train,
              eval_set=(X_val_scaled, y_val),
              early_stopping_rounds=10)
```

**2. Ensemble de Modelos (Stacking)**

Combinar múltiplos modelos para melhor performance:

```python
from sklearn.ensemble import StackingClassifier

# Definir modelos base
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=42)),
    ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42))
]

# Modelo meta (combina predições dos modelos base)
meta_model = LogisticRegression(random_state=42)

# Stacking ensemble
stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stacking_model.fit(X_train_scaled, y_train)
y_pred = stacking_model.predict(X_test_scaled)

print(f"F1-Score do Ensemble: {f1_score(y_test, y_pred):.4f}")
```

**3. Calibração de Probabilidades**

Melhorar a confiabilidade das probabilidades preditas:

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar modelo
calibrated_model = CalibratedClassifierCV(
    best_model, 
    method='isotonic',  # ou 'sigmoid'
    cv=5
)

calibrated_model.fit(X_train_scaled, y_train)

# Comparar probabilidades antes e depois
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Antes da calibração
prob_pos_original = best_model.predict_proba(X_test_scaled)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, prob_pos_original, n_bins=10
)
ax[0].plot(mean_predicted_value, fraction_of_positives, 's-', label='Original')
ax[0].plot([0, 1], [0, 1], 'k--', label='Perfeitamente calibrado')
ax[0].set_title('Antes da Calibração')
ax[0].set_xlabel('Probabilidade Média Predita')
ax[0].set_ylabel('Fração de Positivos')
ax[0].legend()

# Depois da calibração
prob_pos_calibrated = calibrated_model.predict_proba(X_test_scaled)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(
    y_test, prob_pos_calibrated, n_bins=10
)
ax[1].plot(mean_predicted_value, fraction_of_positives, 's-', label='Calibrado')
ax[1].plot([0, 1], [0, 1], 'k--', label='Perfeitamente calibrado')
ax[1].set_title('Depois da Calibração')
ax[1].set_xlabel('Probabilidade Média Predita')
ax[1].set_ylabel('Fração de Positivos')
ax[1].legend()

plt.tight_layout()
plt.show()
```

**4. Análise de Subgrupos**

Avaliar performance em diferentes grupos demográficos:

```python
def analyze_model_by_subgroups(df, predictions, target):
    """Analisar performance do modelo por subgrupos"""
    
    df_analysis = df.copy()
    df_analysis['prediction'] = predictions
    df_analysis['target'] = target
    
    # Análise por gênero
    print("Performance por Gênero:")
    for gender in df_analysis['gender'].unique():
        mask = df_analysis['gender'] == gender
        f1 = f1_score(df_analysis.loc[mask, 'target'], 
                     df_analysis.loc[mask, 'prediction'])
        print(f"  {gender}: F1-Score = {f1:.4f}")
    
    # Análise por faixa etária
    print("\nPerformance por Faixa Etária:")
    age_bins = [0, 30, 45, 60, 120]
    age_labels = ['<30', '30-45', '45-60', '60+']
    df_analysis['age_group'] = pd.cut(df_analysis['age'], bins=age_bins, labels=age_labels)
    
    for age_group in age_labels:
        mask = df_analysis['age_group'] == age_group
        if mask.sum() > 0:
            f1 = f1_score(df_analysis.loc[mask, 'target'], 
                         df_analysis.loc[mask, 'prediction'])
            print(f"  {age_group}: F1-Score = {f1:.4f}")
    
    # Análise por região
    print("\nPerformance por Região:")
    for region in df_analysis['region'].unique():
        mask = df_analysis['region'] == region
        f1 = f1_score(df_analysis.loc[mask, 'target'], 
                     df_analysis.loc[mask, 'prediction'])
        print(f"  {region}: F1-Score = {f1:.4f}")

# Executar análise
analyze_model_by_subgroups(df_test, y_test_pred, y_test)
```

### 12.3 Melhorias de Longo Prazo (3-6 meses)

**1. Deep Learning com Redes Neurais**

Para datasets muito grandes ou padrões muito complexos:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_neural_network(input_dim):
    """Criar rede neural para classificação"""
    
    model = keras.Sequential([
        # Camada de entrada + primeira hidden layer
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Camada de saída
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model

# Criar e treinar
nn_model = create_neural_network(X_train_scaled.shape[1])

history = nn_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ],
    verbose=1
)

# Avaliar
y_pred_proba_nn = nn_model.predict(X_test_scaled).flatten()
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int)

print(f"Neural Network F1-Score: {f1_score(y_test, y_pred_nn):.4f}")
```

**2. AutoML para Automação**

Usar AutoML para explorar automaticamente arquiteturas e hiperparâmetros:

```python
# H2O AutoML
import h2o
from h2o.automl import H2OAutoML

h2o.init()

# Converter dados para H2O Frame
train_h2o = h2o.H2OFrame(pd.concat([X_train_scaled_df, y_train_df], axis=1))
test_h2o = h2o.H2OFrame(pd.concat([X_test_scaled_df, y_test_df], axis=1))

# Definir variáveis
x = train_h2o.columns
y = 'heart_attack'
x.remove(y)

# Executar AutoML
aml = H2OAutoML(max_runtime_secs=3600, seed=42, max_models=20)
aml.train(x=x, y=y, training_frame=train_h2o)

# Leaderboard (ranking de modelos)
lb = aml.leaderboard
print(lb.head(rows=lb.nrows))

# Melhor modelo
best_model_h2o = aml.leader
perf = best_model_h2o.model_performance(test_h2o)
print(perf)
```

**Alternativa - PyCaret:**

```python
from pycaret.classification import *

# Setup
clf = setup(
    data=train_df,
    target='heart_attack',
    session_id=42,
    normalize=True,
    transformation=False,
    ignore_low_variance=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9
)

# Comparar todos os modelos
best_models = compare_models(n_select=5, sort='F1')

# Tunar o melhor
tuned_model = tune_model(best_models[0], optimize='F1')

# Ensemble
blended_model = blend_models(best_models, optimize='F1')

# Avaliar
evaluate_model(blended_model)
```

**3. Deployment e Monitoramento**

Colocar o modelo em produção:

```python
# Salvar modelo
import joblib

joblib.dump(best_model, 'heart_attack_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# API Flask para servir o modelo
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar modelo
model = joblib.load('heart_attack_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('label_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint de predição"""
    try:
        # Receber dados
        data = request.get_json()
        
        # Pré-processar
        df_input = pd.DataFrame([data])
        
        # Codificar categóricas
        for col, encoder in encoders.items():
            if col in df_input.columns:
                df_input[col] = encoder.transform(df_input[col])
        
        # Normalizar
        X_scaled = scaler.transform(df_input)
        
        # Predizer
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]
        
        # Retornar resultado
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'risk_level': 'HIGH' if probability > 0.5 else 'LOW'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```

**4. Monitoramento de Data Drift**

Detectar quando os dados em produção divergem dos dados de treino:

```python
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, production_data, threshold=0.05):
    """Detectar drift usando teste Kolmogorov-Smirnov"""
    
    drift_report = []
    
    for column in reference_data.columns:
        if reference_data[column].dtype in ['int64', 'float64']:
            # Teste KS
            statistic, p_value = ks_2samp(
                reference_data[column],
                production_data[column]
            )
            
            drift_detected = p_value < threshold
            
            drift_report.append({
                'feature': column,
                'p_value': p_value,
                'drift_detected': drift_detected
            })
    
    return pd.DataFrame(drift_report)

# Uso
drift_df = detect_data_drift(X_train_scaled_df, X_production_scaled_df)
print(drift_df[drift_df['drift_detected']])
```

### 12.4 Considerações Éticas e de Implementação

**1. Fairness e Viés**

```python
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Análise de disparate impact
# Verificar se o modelo discrimina baseado em atributos protegidos (gênero, raça, etc.)

def analyze_fairness(X, y, y_pred, sensitive_feature):
    """Analisar fairness do modelo"""
    
    df_fairness = pd.DataFrame({
        'target': y,
        'prediction': y_pred,
        'sensitive': X[sensitive_feature]
    })
    
    print(f"Análise de Fairness para: {sensitive_feature}")
    print("="*60)
    
    for group in df_fairness['sensitive'].unique():
        mask = df_fairness['sensitive'] == group
        
        # Métricas por grupo
        accuracy = accuracy_score(df_fairness.loc[mask, 'target'], 
                                 df_fairness.loc[mask, 'prediction'])
        precision = precision_score(df_fairness.loc[mask, 'target'], 
                                   df_fairness.loc[mask, 'prediction'])
        recall = recall_score(df_fairness.loc[mask, 'target'], 
                            df_fairness.loc[mask, 'prediction'])
        
        print(f"\nGrupo: {group}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
    
    # Calcular Disparate Impact
    # (Taxa de predições positivas para grupo desfavorecido) / 
    # (Taxa de predições positivas para grupo favorecido)
    # Deve estar entre 0.8 e 1.25 para ser considerado justo

analyze_fairness(X_test_df, y_test, y_pred, 'gender')
analyze_fairness(X_test_df, y_test, y_pred, 'region')
```

**2. Explicabilidade Individual com LIME**

```python
import lime
import lime.lime_tabular

# Criar explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled,
    feature_names=feature_names,
    class_names=['Sem Ataque', 'Com Ataque'],
    mode='classification'
)

# Explicar predição de um paciente específico
i = 0  # Índice do paciente
exp = explainer.explain_instance(
    X_test_scaled[i], 
    best_model.predict_proba,
    num_features=10
)

# Visualizar
exp.show_in_notebook(show_table=True)
exp.as_pyplot_figure()
plt.title(f'Explicação LIME para Paciente {i}')
plt.tight_layout()
plt.show()

# Salvar explicação
exp.save_to_file(f'lime_explanation_patient_{i}.html')
```

**3. Documentação e Governança**

Criar documentação completa seguindo o framework Model Cards:

```python
model_card = {
    "model_details": {
        "name": "Heart Attack Prediction Model",
        "version": "1.0.0",
        "date": "2024-01-15",
        "type": "Random Forest Classifier",
        "authors": ["Equipe Data Science"]
    },
    
    "intended_use": {
        "primary_use": "Triagem e identificação de pacientes de alto risco para ataque cardíaco",
        "out_of_scope": "Diagnóstico definitivo, substituição de avaliação médica"
    },
    
    "performance": {
        "f1_score": 0.85,
        "precision": 0.83,
        "recall": 0.87,
        "roc_auc": 0.91
    },
    
    "limitations": [
        "Treinado apenas em dados da Indonésia - generalização para outras populações não garantida",
        "Não captura fatores genéticos e histórico familiar detalhado",
        "Requer atualização periódica com novos dados"
    ],
    
    "fairness_considerations": [
        "Avaliar periodicamente disparate impact entre grupos demográficos",
        "Monitorar taxa de falsos negativos em grupos vulneráveis"
    ]
}

# Salvar
import json
with open('model_card.json', 'w') as f:
    json.dump(model_card, f, indent=2)

    ### 11.2 Estratégia de Feature Engineering

**Oportunidades Implementadas:**

1. **Flag de Valores Ausentes (`alcohol_missing`)**
   - Captura padrão de missingness que pode ser informativo
   - Exemplo: Pessoas que não informam consumo de álcool podem ter padrão de comportamento específico

**Oportunidades Futuras:**

1. **Índices Clínicos Derivados:**
   ```python
   # IMC (Índice de Massa Corporal)
   df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
   
   # Razão Colesterol Total/HDL (indicador de risco)
   df['cholesterol_ratio'] = df['cholesterol_level'] / df['cholesterol_hdl']
   
   # Pressão Arterial Média
   df['mean_arterial_pressure'] = (df['blood_pressure_systolic'] + 2 * df['blood_pressure_diastolic']) / 3
   
   # Índice de Risco Cardiovascular Combinado
   df['risk_score'] = (df['age'] * 0.2 + df['cholesterol_level'] * 0.3 + 
                       df['blood_pressure_systolic'] * 0.3 + df['obesity'] * 20)
   ```

2. **Binning de Variáveis Contínuas:**
   ```python
   # Categorizar idade em faixas de risco
   df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 30, 45, 60, 100], 
                            labels=['Jovem', 'Meia-idade', 'Idoso', 'Muito Idoso'])
   
   # Categorizar níveis de colesterol segundo guidelines médicos
   df['cholesterol_category'] = pd.cut(df['cholesterol_level'],
                                       bins=[0, 200, 240, 1000],
                                       labels=['Desejável', 'Limítrofe', 'Alto'])
   ```

3. **Interações entre Features:**
   ```python
   # Interação diabetes × hipertensão (comorbidades)
   df['diabetes_hypertension'] = df['diabetes'] * df['hypertension']
   
   # Interação fumante × idade (risco acumulado)
   df['smoking_age_interaction'] = df['smoking_current'] * df['age']
   
   # Score de estilo de vida
   df['lifestyle_score'] = (df['physical_activity_low'] + df['dietary_habits_unhealthy'] + 
                            df['smoking_current'] + df['stress_high'])
   ```


   # 11. Abordagens Metodológicas Detalhadas e Justificativas

Nesta seção, apresentamos uma análise detalhada das decisões metodológicas adotadas ao longo do projeto, suas justificativas e alternativas consideradas.

### 11.1 Estratégia de Limpeza de Dados

**Decisões Tomadas:**

1. **Imputação de Valores Ausentes com Moda + Flag Indicadora**
   
   **Justificativa:**
   - A variável `alcohol_consumption` tinha 59,9% de valores ausentes
   - Remover a coluna resultaria em perda de informação potencialmente valiosa
   - Remover as linhas eliminaria quase 60% do dataset
   - A imputação com moda preserva a distribuição original
   - A flag indicadora permite que o modelo aprenda se a ausência é informativa
   
   **Alternativas consideradas:**
   - Imputação por MICE (Multiple Imputation by Chained Equations): mais complexa, mas poderia capturar relações entre variáveis
   - Criar categoria "Unknown": semanticamente clara, mas aumenta cardinalidade
   - Deep learning imputation: requer mais dados e recursos computacionais

2. **Manutenção de Outliers Clínicos**
   
   **Justificativa:**
   - Em dados médicos, valores extremos podem ser clinicamente significativos
   - Um paciente com colesterol de 400 mg/dL é um caso válido de alto risco
   - Remoção automática poderia eliminar exatamente os casos mais críticos
   - Modelos baseados em árvore (Random Forest) são robustos a outliers
   
   **Quando remover outliers:**
   - Valores biologicamente impossíveis (idade negativa, pressão arterial = 0)
   - Erros claros de digitação (altura = 999 cm)
   - Após validação com especialistas de domínio



### 11.3 Estratégia de Modelagem

**Por que múltiplos modelos?**

Testamos 4 algoritmos diferentes porque:
1. **Não há algoritmo universalmente melhor**: O teorema "No Free Lunch" diz que nenhum algoritmo é superior em todos os problemas
2. **Características diferentes**: Cada algoritmo captura padrões diferentes nos dados
3. **Comparação objetiva**: Permite escolher baseado em métricas, não em suposições

**Análise de cada modelo:**

| Modelo | Vantagens | Desvantagens | Quando usar |
|--------|-----------|--------------|-------------|
| Regressão Logística | - Rápido<br>- Interpretável<br>- Bom baseline | - Assume linearidade<br>- Não captura interações complexas | - Relações lineares<br>- Necessidade de interpretabilidade |
| Árvore de Decisão | - Muito interpretável<br>- Captura não-linearidades<br>- Não requer normalização | - Tende a overfitting<br>- Instável | - Exploração inicial<br>- Regras de decisão simples |
| Random Forest | - Robusto<br>- Lida com não-linearidades<br>- Feature importance | - "Black box"<br>- Computacionalmente custoso | - Melhor performance<br>- Dados complexos |
| KNN | - Simples<br>- Não paramétrico | - Sensível à escala<br>- Lento em grandes datasets<br>- Curse of dimensionality | - Datasets pequenos<br>- Padrões locais |

**Modelos não testados e por quê considerar:**

1. **XGBoost / LightGBM:**
   - Geralmente superam Random Forest
   - Mais eficientes computacionalmente
   - Excelentes em competições de ML
   
2. **Redes Neurais:**
   - Capturam padrões muito complexos
   - Requerem mais dados e tuning
   - Menos interpretáveis
   
3. **SVM (Support Vector Machine):**
   - Excelente para classificação binária
   - Kernel trick captura não-linearidades
   - Lento em grandes datasets

### 11.4 Estratégia de Validação

**Por que 70-15-15?**

- **70% Treino**: Dados suficientes para o modelo aprender padrões robustos
- **15% Validação**: Comparar modelos e ajustar hiperparâmetros sem tocar no teste
- **15% Teste**: Avaliação final imparcial

**Alternativas:**

1. **K-Fold Cross-Validation:**
   ```python
   from sklearn.model_selection import cross_val_score
   
   scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
   print(f"F1-Score médio: {scores.mean():.4f} (+/- {scores.std():.4f})")
   ```
   - **Vantagem**: Uso mais eficiente dos dados
   - **Desvantagem**: 5x mais custoso computacionalmente

2. **Stratified K-Fold:**
   - Mantém proporção de classes em cada fold
   - Essencial em datasets desbalanceados

**Métricas de Avaliação:**

**Por que F1-Score como métrica principal?**

Em contexto médico, precisamos balancear:
- **Precision (Precisão)**: Evitar alarmes falsos → custos desnecessários, ansiedade
- **Recall (Sensibilidade)**: Detectar todos os casos reais → salvar vidas

**Tabela de Interpretação:**

| Métrica | Pergunta que responde | Importância Clínica |
|---------|----------------------|---------------------|
| Accuracy | Quantas predições estão corretas? | Baixa (enganosa em desbalanceamento) |
| Precision | Dos positivos preditos, quantos são verdadeiros? | Média (custo de falsos positivos) |
| Recall | Dos positivos reais, quantos detectamos? | ALTA (custo de falsos negativos = vidas) |
| F1-Score | Balanço entre Precision e Recall | ALTA (métrica principal) |
| ROC-AUC | Capacidade de separação em diferentes thresholds | Alta (comparação de modelos) |

**Trade-off Precision vs Recall:**

```python
# Ajustar threshold para priorizar Recall (detectar mais casos)
y_pred_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Mais conservador que 0.5
y_pred_custom = (y_pred_proba >= threshold).astype(int)

# Isso aumenta Recall mas diminui Precision
# Útil em triagem onde é melhor ter falsos positivos que falsos negativos
```

### 11.5 Tratamento de Desbalanceamento de Classes

**Se o dataset estiver desbalanceado (ex: 80% sem ataque, 20% com ataque):**

**Técnicas disponíveis:**

1. **Class Weights:**
   ```python
   from sklearn.utils.class_weight import compute_class_weight
   
   class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_train), 
                                        y=y_train)
   
   model = RandomForestClassifier(class_weight='balanced')
   ```
   - **Vantagem**: Simples, não modifica dados
   - **Funcionamento**: Penaliza mais erros na classe minoritária

2. **SMOTE (Synthetic Minority Over-sampling Technique):**
   ```python
   from imblearn.over_sampling import SMOTE
   
   smote = SMOTE(random_state=42)
   X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
   ```
   - **Vantagem**: Aumenta classe minoritária sinteticamente
   - **Desvantagem**: Pode criar padrões artificiais

3. **Undersampling:**
   ```python
   from imblearn.under_sampling import RandomUnderSampler
   
   rus = RandomUnderSampler(random_state=42)
   X_train_balanced, y_train_balanced = rus.fit_resample(X_train, y_train)
   ```
   - **Vantagem**: Rápido, balanceia classes
   - **Desvantagem**: Perde informação da classe majoritária

4. **Ensemble de técnicas:**
   ```python
   from imblearn.combine import SMOTETomek
   
   smt = SMOTETomek(random_state=42)
   X_train_balanced, y_train_balanced = smt.fit_resample(X_train, y_train)
   ```