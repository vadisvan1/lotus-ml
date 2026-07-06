# app.py - LÓTUS ML Mobile (Streamlit)
import streamlit as st
import numpy as np
import pandas as pd
from collections import Counter
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="LÓTUS ML Mobile", layout="centered")
st.title("🌸 LÓTUS ML — Lottery Intelligence")
st.caption("Otimizado para celular | Processamento em nuvem | Jogue com responsabilidade")

# Session State
if "historico" not in st.session_state:
    st.session_state.historico = []
if "modelo" not in st.session_state:
    st.session_state.modelo = None
if "scaler" not in st.session_state:
    st.session_state.scaler = StandardScaler()
if "treinado" not in st.session_state:
    st.session_state.treinado = False

NUMEROS_MAX = 60

def criar_features(historico, janela=5):
    X, y = [], []
    for i in range(janela, len(historico)):
        janela_dados = historico[i-janela:i]
        proximo = historico[i]
        todos = [n for j in janela_dados for n in j]
        freq = Counter(todos)
        
        features = []
        for n in range(1, NUMEROS_MAX+1):
            features.append([
                freq.get(n, 0),
                sum(1 for j in janela_dados if n in j),
                np.mean([j[2] for j in janela_dados]),
                np.std(todos),
                sum(1 for j in janela_dados if any(abs(a-n)==1 for a in j)),
                len([x for x in todos if x%2==0]),
                sum(janela_dados[-1])/6
            ])
        X.append(features)
        y.append([1 if n in proximo else 0 for n in range(1, NUMEROS_MAX+1)])
    return np.array(X), np.array(y)

def treinar_modelo(historico, epochs=50):
    if len(historico) < 30:
        st.error("❌ Mínimo de 30 concursos necessários para treinar.")
        return False
    
    with st.spinner("🔄 Criando features e treinando rede neural..."):
        X, y = criar_features(historico)
        X = X.reshape(X.shape[0], -1)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.scaler.fit(X_train)
        X_train = st.session_state.scaler.transform(X_train)
        X_test = st.session_state.scaler.transform(X_test)
        
        model = Sequential([
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(), Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(NUMEROS_MAX, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=epochs, batch_size=32,
                  validation_data=(X_test, y_test), callbacks=[early], verbose=0)
        
        st.session_state.modelo = model
        st.session_state.treinado = True
        st.success("✅ Modelo treinado e salvo na sessão!")
        return True

def gerar_dicas(qtd=5, temperatura=0.8):
    if not st.session_state.treinado:
        st.warning("⚠️ Modelo não treinado. Treine primeiro ou carregue dados.")
        return
    
    ultimos = st.session_state.historico[-5:]
    todos = [n for j in ultimos for n in j]
    freq = Counter(todos)
    
    features = []
    for n in range(1, NUMEROS_MAX+1):
        features.append([
            freq.get(n, 0),
            sum(1 for j in ultimos if n in j),
            np.mean([j[2] for j in ultimos]),
            np.std(todos),
            sum(1 for j in ultimos if any(abs(a-n)==1 for a in j)),
            len([x for x in todos if x%2==0]),
            sum(ultimos[-1])/6
        ])
    
    X_pred = np.array(features).reshape(1, -1)
    X_pred = st.session_state.scaler.transform(X_pred)
    probs = st.session_state.modelo.predict(X_pred, verbose=0)[0]
    
    probs = np.power(probs, 1/temperatura)
    probs = probs / probs.sum()
    
    resultados = []
    for _ in range(qtd):
        nums = []
        indices = list(range(NUMEROS_MAX))
        p_temp = probs.copy()
        while len(nums) < 6:
            esc = random.choices(indices, weights=p_temp[indices], k=1)[0]
            nums.append(esc+1)
            indices.remove(esc)
            p_temp[esc] *= 0.2
        resultados.append(sorted(nums))
    
    df = pd.DataFrame(resultados, columns=[f"D{n+1}" for n in range(6)])
    df["Soma"] = df.sum(axis=1)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("💡 Combinações geradas com viés estatístico + rede neural. Loterias são aleatórias por natureza.")

# ================= UI =================
st.divider()
st.subheader("📥 Carregar Histórico (Mega-Sena)")
st.caption("Cole os resultados, um por linha. Ex: `5 12 23 34 45 57`")
texto_historico = st.text_area("Resultados passados:", height=150)

if st.button("✅ Carregar e Treinar Modelo", type="primary"):
    linhas = [l.strip() for l in texto_historico.split("\n") if l.strip()]
    historico = []
    for l in linhas:
        try:
            nums = list(map(int, l.split()))
            if len(nums) == 6:
                historico.append(sorted(nums))
        except: continue
    if len(historico) < 30:
        st.error(f"❌ Apenas {len(historico)} jogos válidos encontrados. Mínimo: 30.")
    else:
        st.session_state.historico = historico
        treinar_modelo(historico, epochs=50)

st.divider()
st.subheader("🎲 Gerar Dicas")
col1, col2 = st.columns(2)
with col1:
    qtd = st.number_input("Quantidade", min_value=1, max_value=20, value=6)
with col2:
    temp = st.slider("Temperatura (criatividade)", 0.5, 1.5, 0.8, 0.05)

if st.button("🚀 Gerar Combinações", type="primary"):
    if not st.session_state.treinado:
        st.warning("⚠️ Treine o modelo primeiro carregando o histórico.")
    else:
        gerar_dicas(qtd, temp)

st.divider()
st.caption("🔒 Nenhum dado é enviado para terceiros. Processamento ocorre no servidor Streamlit.")
st.caption("⚠️ Aviso legal: Este sistema não garante premiações. Loterias são jogos de azar com eventos independentes. Jogue com consciência.")
