import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Probabilidad y Estadistica", layout="wide")

st.markdown("""
<style>
.stApp { background:#0a0e1a; color:#e8eaf0; }
[data-testid="stSidebar"] { background:#0f1629 !important; }
.card { background:#111827; border:1px solid #2a3f6f; border-radius:10px;
        padding:18px; text-align:center; }
.val  { font-size:2rem; font-weight:700; color:#4fc3f7; font-family:monospace; }
.lbl  { font-size:0.75rem; color:#8899bb; text-transform:uppercase; letter-spacing:1px; }
.box  { background:#0f1e38; border:1px solid #1e3a6e; border-radius:8px;
        padding:14px; font-family:monospace; color:#90caf9; margin:10px 0; }
</style>
""", unsafe_allow_html=True)

LAYOUT = dict(paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
              font=dict(color="#c0ccdd"), margin=dict(l=40,r=20,t=50,b=40))

def card(v, l):
    st.markdown(f'<div class="card"><div class="val">{v}</div><div class="lbl">{l}</div></div>', unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Probabilidad y Estadistica 2A")
    f = st.file_uploader("Cargar CSV", type="csv")

if not f:
    st.info("Sube un archivo CSV desde el panel lateral para comenzar.")
    st.stop()

df = pd.read_csv(f)

# ── DETECCIÓN AUTOMÁTICA ──────────────────────────────────────────────────────
num_cols, cat_cols, date_cols, bin_cols = [], [], [], []
for col in df.columns:
    s = df[col].dropna()
    if s.dtype == "object":
        try:
            pd.to_datetime(s.head(30)); date_cols.append(col); continue
        except: pass
    if pd.api.types.is_numeric_dtype(s):
        (bin_cols if set(s.unique()) <= {0,1} else num_cols).append(col)
    else:
        (bin_cols if {str(v).lower() for v in s.unique()} <= {"si","sí","no","yes","true","false","0","1"} else cat_cols).append(col)

# ── OVERVIEW ──────────────────────────────────────────────────────────────────
st.markdown("## Dataset")
c1,c2,c3,c4 = st.columns(4)
with c1: card(f"{len(df):,}", "Filas")
with c2: card(len(df.columns), "Columnas")
with c3: card(df.isnull().sum().sum(), "Nulos")
with c4: card(len(num_cols), "Numéricas")

tags = "".join([
    *[f'<span style="background:#0d2137;color:#4fc3f7;border:1px solid #1e5f8a;padding:2px 9px;border-radius:20px;font-size:.72rem;margin:2px;display:inline-block"> {c}</span>' for c in num_cols],
    *[f'<span style="background:#1a0d37;color:#ce93d8;border:1px solid #6a3a8a;padding:2px 9px;border-radius:20px;font-size:.72rem;margin:2px;display:inline-block"> {c}</span>' for c in cat_cols],
    *[f'<span style="background:#0d2e1a;color:#66bb6a;border:1px solid #2a6a3a;padding:2px 9px;border-radius:20px;font-size:.72rem;margin:2px;display:inline-block"> {c}</span>' for c in date_cols],
    *[f'<span style="background:#2e1a0d;color:#ffb74d;border:1px solid #8a5a2a;padding:2px 9px;border-radius:20px;font-size:.72rem;margin:2px;display:inline-block"> {c}</span>' for c in bin_cols],
])
st.markdown(tags, unsafe_allow_html=True)

with st.expander("Ver datos"): st.dataframe(df.head(40), use_container_width=True)

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
st.markdown("## Configuración")
all_cols = bin_cols + cat_cols + num_cols
ca, cb = st.columns(2)
target   = ca.selectbox("Variable objetivo", all_cols)
cond_col = cb.selectbox("Variable de condición", [c for c in num_cols + bin_cols if c != target])

threshold = None
if cond_col in num_cols:
    threshold = st.slider(f"Umbral para '{cond_col}'",
                          float(df[cond_col].min()), float(df[cond_col].max()),
                          float(df[cond_col].quantile(0.75)))

features = st.multiselect("Características para Naive Bayes",
                           [c for c in num_cols + bin_cols if c != target],
                           default=[c for c in num_cols + bin_cols if c != target][:5])

if not st.button("REALIZARº ANÁLISIS"): st.stop()

# ── PREPARAR TARGET ───────────────────────────────────────────────────────────
le = LabelEncoder()
df["_t"] = le.fit_transform(df[target].astype(str))
df["_t"] = (df["_t"] == df["_t"].max()).astype(int)

# ── BAYES ─────────────────────────────────────────────────────────────────────
st.markdown("## Teorema de Bayes")
A = df["_t"] == 1
B = (df[cond_col] > threshold) if threshold is not None else df[cond_col].astype(str).str.lower().isin(["si","sí","yes","1","true"])
PA  = A.mean()
PB  = B.mean()
PBA = (A & B).sum() / max(A.sum(), 1)
PAB = (PBA * PA) / max(PB, 1e-9)

c1,c2,c3,c4 = st.columns(4)
with c1: card(f"{PA:.4f}", "P(A)")
with c2: card(f"{PB:.4f}", "P(B)")
with c3: card(f"{PBA:.4f}", "P(B|A)")
with c4: card(f"{PAB:.4f}", "P(A|B) Bayes")

st.markdown(f"""<div class="box">
P(A|B) = P(B|A) × P(A) / P(B) = {PBA:.4f} × {PA:.4f} / {PB:.4f} = <b style="color:#4fc3f7">{PAB:.4f}</b><br>
{'La evidencia AUMENTA el riesgo' if PAB > PA else 'La evidencia REDUCE el riesgo'} &nbsp;|&nbsp; Cambio: {(PAB-PA)/PA*100:+.1f}%
</div>""", unsafe_allow_html=True)

# ── NAIVE BAYES ───────────────────────────────────────────────────────────────
cm_data = acc = sens = spec = y_true = y_prob = None
if features:
    st.markdown("## Naive Bayes")
    try:
        X = df[features].fillna(0)
        for col in X.select_dtypes("object"): X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X_tr,X_te,y_tr,y_te = train_test_split(X, df["_t"], test_size=0.3, random_state=42)
        clf = GaussianNB().fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:,1]
        y_true = y_te.values
        cm_data = confusion_matrix(y_te, y_pred)
        acc  = accuracy_score(y_te, y_pred)
        tn,fp,fn,tp = cm_data.ravel()
        sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1)
        c1,c2,c3 = st.columns(3)
        with c1: card(f"{acc*100:.1f}%","Exactitud")
        with c2: card(f"{sens*100:.1f}%","Sensibilidad")
        with c3: card(f"{spec*100:.1f}%","Especificidad")
    except Exception as e:
        st.warning(f"Error Naive Bayes: {e}")

# ── VISUALIZACIONES ───────────────────────────────────────────────────────────
st.markdown("## Visualizaciones")
t1,t2,t3,t4,t5 = st.tabs(["Histogramas","Serie Temporal","Curva Posterior","Confusión","Comparación"])

with t1:
    for i,col in enumerate(num_cols[:6]):
        fig = px.histogram(df, x=col, nbins=30, color_discrete_sequence=["#4fc3f7"])
        fig.update_layout(**LAYOUT, title=col, height=280)
        st.plotly_chart(fig, use_container_width=True)

with t2:
    if date_cols and num_cols:
        dc = st.selectbox("Fecha", date_cols); yc = st.selectbox("Variable Y", num_cols[:5])
        try:
            tmp = df[[dc,yc,"_t"]].copy(); tmp[dc] = pd.to_datetime(tmp[dc],errors="coerce")
            tmp = tmp.dropna().sort_values(dc)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tmp[tmp["_t"]==0][dc],y=tmp[tmp["_t"]==0][yc],mode="lines",name="Normal",line=dict(color="#4fc3f7")))
            fig.add_trace(go.Scatter(x=tmp[tmp["_t"]==1][dc],y=tmp[tmp["_t"]==1][yc],mode="markers",name="Evento",marker=dict(color="#ffb74d",size=9,symbol="x")))
            fig.update_layout(**LAYOUT, title=f"Serie: {yc}", height=400)
            st.plotly_chart(fig, use_container_width=True)
        except: st.warning("Error al graficar serie temporal.")
    else: st.info("Necesitas columnas de fecha y numéricas.")

with t3:
    if cond_col in num_cols:
        thrs = np.linspace(df[cond_col].quantile(.1), df[cond_col].quantile(.9), 50)
        pab_list = []
        for thr in thrs:
            b_ = df[cond_col] > thr
            pb_ = b_.mean(); pba_ = (A&b_).sum()/max(A.sum(),1)
            pab_list.append((pba_*PA)/max(pb_,1e-9))
        fig = go.Figure()
        fig.add_hline(y=PA, line=dict(color="#ffb74d",dash="dash"), annotation_text=f"P(A)={PA:.3f}", annotation_font_color="#ffb74d")
        fig.add_trace(go.Scatter(x=thrs,y=pab_list,name="P(A|B)",line=dict(color="#4fc3f7",width=3),fill="tozeroy",fillcolor="rgba(79,195,247,0.1)"))
        if threshold: fig.add_vline(x=threshold,line=dict(color="#66bb6a",dash="dash"))
        fig.update_layout(**LAYOUT,title="Probabilidad Posterior vs Umbral",xaxis_title=cond_col,yaxis_title="P(A|B)",height=400)
        st.plotly_chart(fig, use_container_width=True)

with t4:
    if cm_data is not None:
        fig = px.imshow(cm_data,text_auto=True,x=["Neg","Pos"],y=["Neg","Pos"],
                        color_continuous_scale=[[0,"#0a0e1a"],[1,"#4fc3f7"]],title="Matriz de Confusión")
        fig.update_layout(**LAYOUT,height=380)
        st.plotly_chart(fig, use_container_width=True)
    else: st.info("Ejecuta Naive Bayes para ver la matriz.")

with t5:
    fig = go.Figure(go.Bar(
        x=["P(A) — Sin evidencia","P(A|B) — Con Bayes"], y=[PA, PAB],
        marker_color=["#1565c0","#4fc3f7"],
        text=[f"{PA:.4f}",f"{PAB:.4f}"], textposition="outside",
        textfont=dict(size=16, color="white"), width=0.4
    ))
    fig.update_layout(**LAYOUT, title="Impacto de la Evidencia", yaxis_range=[0, max(PA,PAB)*1.4], height=380)
    st.plotly_chart(fig, use_container_width=True)

# ── INSIGHTS ──────────────────────────────────────────────────────────────────
st.markdown("## Insights")
cambio = (PAB-PA)/PA*100
msgs = [
    f"{' Riesgo AUMENTA' if PAB>PA else ' Riesgo REDUCE'}: la evidencia cambia P(A) de {PA:.3f} → {PAB:.3f} ({cambio:+.1f}%)",
    f"Tasa base del evento: {PA*100:.2f}% del total de registros.",
    f"Dado el evento, la condición se cumple el {PBA*100:.1f}% de las veces."
]
if acc: msgs.append(f" Naive Bayes: {acc*100:.1f}% exactitud, sensibilidad {sens*100:.1f}%, especificidad {spec*100:.1f}%")
for m in msgs:

    st.markdown(f'<div style="background:#0d1f0d;border:1px solid #2a5a2a;border-radius:8px;padding:14px;margin:6px 0;color:#a5d6a7">{m}</div>', unsafe_allow_html=True)
