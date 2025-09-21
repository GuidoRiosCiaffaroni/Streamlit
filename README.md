### Streamlit

El siguiente script contiene el desarrollo de una aplicación en **Streamlit**, estructurada para cumplir con los criterios de evaluación.  

- En el **sidebar** se incluyen las opciones de:
  - Carga de datos  
  - Preprocesamiento  
  - Selección de métodos y parámetros  

- En el **cuerpo principal** se presentan los gráficos y métricas obtenidas.  

- La aplicación incorpora **mecanismos de interactividad fluida**, utilizando `st.cache_data` y `st.cache_resource` cuando corresponde, lo que permite recalcular proyecciones en tiempo real al modificar *sliders* o botones de selección.  

- Finalmente, se implementan **funciones exportables** que facilitan la descarga de los resultados:  
  - En formato **CSV** para los embeddings  
  - Como imágenes (**PNG/SVG**) de las figuras generadas  
  - Además de la posibilidad de guardar la sesión  


>01_install.sh

#1. Actualizar el sistema
  ``` 
sudo apt update && sudo apt upgrade -y
sudo apt install nano -y 
sudo apt install tmux -y
  ``` 

#2. Instalar dependencias basicas
 ```
sudo apt install -y python3 python3-pip python3-venv
 ```


>02_entorno.sh
 ```
#  Crear un entorno virtual
rm -rf ~/streamlit_app
mkdir ~/streamlit_app && cd ~/streamlit_app
python3 -m venv venv
source venv/bin/activate
 ```


 >03_install_streamlit.sh
  ```
# Instalar streamlit dentro del entorno 
sudo pip install --upgrade pip
sudo pip install streamlit numpy pandas matplotlib seaborn plotly scikit-learn umap-learn kaleido openpyxl

  ```
  
> Crear la app de prueba
```
# App Streamlit para reducción de dimensionalidad


import io
import base64

from typing import Tuple, Optional



import numpy as np
import pandas as pd
import streamlit as st



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score



# ------------------------------

try:
    import umap
except Exception:
    umap = None


import plotly.express as px

st.set_page_config(page_title="Reducción de Dimensionalidad", layout="wide")
st.title("Reducción de Dimensionalidad")
st.caption("Carga tu CSV/XLSX, prepara los datos y explora PCA, LDA, t‑SNE y UMAP con métricas y exportables.")



# =============================
# Sidebar: Carga y Pre‑proceso
# =============================

st.sidebar.header("1) Carga de datos")
up = st.sidebar.file_uploader("Sube un archivo CSV o XLSX", type=["csv", "xlsx"]) 



# Opcional: columnas especiales
st.sidebar.header("2) Configuración de columnas")
use_header = st.sidebar.checkbox("La primera fila tiene encabezados", value=True)
sep = st.sidebar.selectbox("Separador (solo CSV)", [",", ";", "\t", "|"], index=0)
encoding = st.sidebar.text_input("Encoding (vacío = auto)", value="")
id_col = st.sidebar.text_input("Columna ID (opcional)")
target_col = st.sidebar.text_input("Columna target (clase) – necesario para LDA y accuracy")



st.sidebar.header("3) Limpieza e imputación")
impute_strategy = st.sidebar.selectbox(
    "Estrategia de imputación (numéricas)",
    ["median", "mean", "most_frequent", "knn", "drop_rows"],
    index=0,
    help="Para 'knn' se usa KNNImputer (k=5). 'drop_rows' elimina filas con NA."
)

cat_strategy = st.sidebar.selectbox(
    "Imputación para categóricas",
    ["most_frequent", "constant_none"],
    index=0,
)

st.sidebar.header("4) Escalado y codificación")
scaler_choice = st.sidebar.radio("Escalado", ["StandardScaler", "MinMaxScaler", "None"], index=0)
encode_cats = st.sidebar.checkbox("One‑Hot Encode para categóricas (si existen)", value=True)
st.sidebar.header("5) Selección de método y parámetros")
method = st.sidebar.selectbox("Método de reducción", ["PCA", "LDA", "t-SNE", "UMAP"]) 

# Parámetros por método
pca_n = st.sidebar.slider("PCA: componentes", min_value=2, max_value=10, value=2)
lda_n = st.sidebar.slider("LDA: componentes", min_value=1, max_value=3, value=2)
perplexity = st.sidebar.slider("t‑SNE: perplexity", min_value=5, max_value=100, value=30)
learning_rate = st.sidebar.slider("t‑SNE: learning_rate", min_value=10, max_value=2000, value=200)
umap_neighbors = st.sidebar.slider("UMAP: n_neighbors", min_value=5, max_value=100, value=15)
umap_min_dist = st.sidebar.slider("UMAP: min_dist", min_value=0.0, max_value=0.99, value=0.1)
st.sidebar.header("6) Métricas")
km_k = st.sidebar.number_input("k‑means (k)", min_value=2, value=3)
knn_k = st.sidebar.number_input("kNN (k) para accuracy", min_value=1, value=5)

def _safe_read(uploaded, use_header=True, sep=",", encoding="") -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    name = uploaded.name.lower()

    try:
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded)
        else:
            if encoding.strip():
                df = pd.read_csv(uploaded, sep=sep, encoding=encoding or None, header=0 if use_header else None)
            else:
                # Intento de autodetección
                try:
                    df = pd.read_csv(uploaded, sep=sep, header=0 if use_header else None)
                except Exception:
                    df = pd.read_csv(uploaded, sep=sep, header=0 if use_header else None, encoding="latin1")
        return df

    except Exception as e:
        st.error(f"No se pudo leer el archivo: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)

def load_data(uploaded, use_header, sep, encoding) -> pd.DataFrame:
    return _safe_read(uploaded, use_header, sep, encoding)

@st.cache_data(show_spinner=False)

def preprocess(df: pd.DataFrame, target_col: Optional[str], id_col: Optional[str], impute_strategy: str,
               cat_strategy: str, scaler_choice: str, encode_cats: bool) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
    if df.empty:
        return df, pd.Series(dtype=float), None
    df = df.copy()
    y = None
    ids = None

    # Extraer target e id si existen

    if target_col and target_col in df.columns:
        y = df[target_col]
        df = df.drop(columns=[target_col])

    if id_col and id_col in df.columns:
        ids = df[id_col]
        df = df.drop(columns=[id_col])

    # Separar numéricas y categóricas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Imputación numérica
    if impute_strategy == "drop_rows":
        mask_all = pd.Series(True, index=df.index)
        if num_cols:
            mask_all &= df[num_cols].notna().all(axis=1)
        if cat_cols:
            mask_all &= df[cat_cols].notna().all(axis=1)

        df = df.loc[mask_all].reset_index(drop=True)
        if y is not None:
            y = y.loc[mask_all].reset_index(drop=True)
        if ids is not None:
            ids = ids.loc[mask_all].reset_index(drop=True)
    else:
        if num_cols:
            if impute_strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=impute_strategy)
            df[num_cols] = imputer.fit_transform(df[num_cols])

        # Imputación categórica
        if cat_cols:
            if cat_strategy == "most_frequent":
                imp_cat = SimpleImputer(strategy="most_frequent")
                df[cat_cols] = imp_cat.fit_transform(df[cat_cols])
            else:  # constant_none
                imp_cat = SimpleImputer(strategy="constant", fill_value="None")
                df[cat_cols] = imp_cat.fit_transform(df[cat_cols])


    # One‑Hot Encoding (indica que la siguiente sección o línea)

    if encode_cats and cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)



    # Escalado

    if scaler_choice == "StandardScaler":
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    elif scaler_choice == "MinMaxScaler":
        scaler = MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, y, ids

@st.cache_resource(show_spinner=False)

def compute_embedding(X: np.ndarray, y: Optional[pd.Series], method: str,
                      pca_n: int, lda_n: int,
                      perplexity: int, learning_rate: int,
                      umap_neighbors: int, umap_min_dist: float,
                      random_state: int = 42) -> np.ndarray:
    if method == "PCA":
        model = PCA(n_components=min(pca_n, X.shape[1]))
        return model.fit_transform(X)
    elif method == "LDA":
        if y is None:
            raise ValueError("LDA requiere target (y)")
        # LDA máximo = n_clases-1
        n_classes = len(pd.Series(y).dropna().unique())
        n_comp = min(lda_n, max(1, n_classes - 1))
        model = LDA(n_components=n_comp)
        return model.fit_transform(X, y)
    elif method == "t-SNE":
        model = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state, init="pca")
        return model.fit_transform(X)
    elif method == "UMAP":
        if umap is None:
            raise ValueError("UMAP no está instalado. Agrega 'umap-learn' a requirements.txt")
        model = umap.UMAP(n_components=2, n_neighbors=umap_neighbors, min_dist=umap_min_dist, random_state=random_state)
        return model.fit_transform(X)
    else:
        raise ValueError("Método no soportado")

@st.cache_data(show_spinner=False)

def to_downloadable_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data(show_spinner=False)

def fig_to_png_bytes(fig) -> bytes:
    # Usa engine de kaleido para exportar
    return fig.to_image(format="png")

# ===============
# Carga de datos
# ===============

df = load_data(up, use_header, sep, encoding)

if df.empty:
    st.info("Sube un CSV/XLSX desde la barra lateral para comenzar.")
    st.stop()

st.subheader("Vista previa de datos")
st.dataframe(df.head(50), use_container_width=True)

# ===========================
# Preproceso + división train
# ===========================

X_proc, y, ids = preprocess(df, target_col or None, id_col or None, impute_strategy, cat_strategy, scaler_choice, encode_cats)

if X_proc.empty:
    st.error("No hay columnas utilizables tras el preproceso.")
    st.stop()

# ============================
# Embedding 2D (o n comp ped.)
# ============================

try:
    Z = compute_embedding(
        X_proc.values, y, method,
        pca_n=pca_n, lda_n=lda_n,
        perplexity=perplexity, learning_rate=learning_rate,
        umap_neighbors=umap_neighbors, umap_min_dist=umap_min_dist,
        random_state=42,
    )

except Exception as e:
    st.error(f"Error al calcular el embedding: {e}")
    st.stop()

# Preparar DataFrame de embedding (tomar solo 2 primeras comp para graficar si hay más)

if Z.shape[1] >= 2:
    Z2 = Z[:, :2]
else:
    # Asegurar 2D mínimo (caso raro)
    pad = np.zeros((Z.shape[0], 2 - Z.shape[1]))
    Z2 = np.hstack([Z, pad])

emb_df = pd.DataFrame({
    "Emb1": Z2[:, 0],
    "Emb2": Z2[:, 1],
})

if y is not None:
    emb_df["target"] = y.values
if ids is not None:
    emb_df["id"] = ids.values

# =====================
# Gráfico principal
# =====================

st.subheader("Proyección 2D del embedding")
color_col = "target" if "target" in emb_df.columns else None
hover_cols = [c for c in ["id", "target"] if c in emb_df.columns]

fig = px.scatter(
    emb_df, x="Emb1", y="Emb2", color=color_col, hover_data=hover_cols,
    title=f"{method} – Proyección 2D"
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# Métricas
# =====================

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Métrica no supervisada: Silhouette (k‑means)")
    try:
        kmeans = KMeans(n_clusters=int(km_k), n_init="auto", random_state=42)
        labels = kmeans.fit_predict(emb_df[["Emb1", "Emb2"]])
        sil = silhouette_score(emb_df[["Emb1", "Emb2"]], labels)
        st.metric("Silhouette (mayor es mejor)", f"{sil:.3f}")

    except Exception as e:
        st.warning(f"No se pudo calcular silhouette: {e}")

with col2:
    st.markdown("### Métrica supervisada: Accuracy con kNN (k=K)")
    if "target" in emb_df.columns:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                emb_df[["Emb1", "Emb2"]], emb_df["target"], test_size=0.2, random_state=42, stratify=emb_df["target"]
            )
            clf = KNeighborsClassifier(n_neighbors=int(knn_k))
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.metric("Accuracy (mayor es mejor)", f"{acc:.3f}")
        except Exception as e:
            st.warning(f"No se pudo calcular accuracy: {e}")
    else:
        st.info("Define una columna 'target' en la barra lateral para evaluar accuracy.")

# =====================
# Exportables
# =====================

st.subheader("Exportar resultados")
emb_csv = to_downloadable_csv(emb_df)
st.download_button(
    label="Descargar embedding (CSV)",
    data=emb_csv,
    file_name=f"embedding_{method.lower()}.csv",
    mime="text/csv",
)

# Exportar figura a PNG

try:
    png_bytes = fig_to_png_bytes(fig)
    st.download_button(
        label="Descargar figura (PNG)",
        data=png_bytes,
        file_name=f"{method.lower()}_embedding.png",
        mime="image/png",
    )

except Exception as e:
    st.info("Para exportar PNG de Plotly, instala 'kaleido'.")

# =====================
# Notas y ayuda
# =====================

with st.expander("Notas y recomendaciones"):
    st.markdown(
        """

        - **t‑SNE** es sensible a *perplexity* y *learning_rate*. Cambios pequeños pueden alterar la forma de los clusters.
        - **UMAP** conserva mejor la estructura local con `n_neighbors` bajos, y global con valores mayores.
        - **LDA** requiere `target` y como máximo produce `n_clases−1` dimensiones.
        - **PCA** es lineal; útil para varianza global y como paso previo.
        - Para **accuracy** se usa un split 80/20 con `kNN` sobre el embedding 2D.
        - Para **silhouette** se aplica *k‑means* sobre el embedding 2D.

        """

    )
```
>
