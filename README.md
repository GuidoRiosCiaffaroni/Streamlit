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
>app.py
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
    st.info("Para exportar PNG de Plotly, usa 'kaleido'.")

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
>Correr el Codigo 
```
streamlit run app.py

```


> Para subir el arcchivo 

```
# pasa la propiedad a tu usuario
sudo chown -R ubuntu:ubuntu /home/ubuntu/Streamlit

```


> ### 1.-
En esta imagen se puede observar que el usuario se encuentra dentro de su entorno virtual (venv) en Ubuntu, específicamente en la carpeta ~/streamlit_app. Desde esa ubicación ejecuta el comando:

Lo anterior indica que el servidor de Streamlit está siendo iniciado correctamente desde el proyecto, constituyendo el paso final tras la configuración del entorno y la instalación de las librerías necesarias.
![000](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/000.png)

> ### 2.-
En esta imagen se puede observar que, tras ejecutar el comando streamlit run app.py dentro del entorno virtual (venv) en la carpeta ~/streamlit_app, el sistema despliega el mensaje de bienvenida de Streamlit.

Se aprecia que la aplicación solicita opcionalmente ingresar un correo electrónico para recibir noticias, ofertas y actualizaciones. Además, informa sobre su política de privacidad y el uso de estadísticas de telemetría, aclarando que no se recopila ni almacena el contenido generado dentro de las aplicaciones (como texto, gráficos o imágenes).

Finalmente, se muestra la instrucción para desactivar la recopilación de estadísticas de uso mediante la configuración en el archivo ~/.streamlit/config.toml, agregando la siguiente línea:
![001](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/001.png)

> ### 3.-
En esta imagen se puede observar que, luego de ejecutar correctamente el comando streamlit run app.py, el sistema indica que la aplicación de Streamlit ya está disponible para visualizarse en el navegador.

Se muestran dos direcciones de acceso:

Local URL: http://localhost:8501, utilizada para acceder directamente desde la misma máquina en que se ejecuta el servidor.

Network URL: http://10.0.2.15:8501, que permite el acceso desde otros dispositivos dentro de la misma red, siempre que el puerto correspondiente se encuentre abierto y accesible.

Este mensaje confirma que el despliegue de la aplicación se realizó de manera exitosa y está lista para ser utilizada.
![002](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/002.png)

> ### 4.-
En esta imagen se puede observar la interfaz inicial de la aplicación desarrollada en Streamlit para la reducción de dimensionalidad.

En el panel lateral izquierdo aparecen las secciones de configuración:

1) Carga de datos, que permite subir un archivo en formato CSV o XLSX mediante un botón de exploración de archivos o arrastrando directamente el fichero, con un límite de 200 MB por archivo.

2) Configuración de columnas, donde se puede indicar si la primera fila contiene encabezados y definir el separador en caso de trabajar con archivos CSV.

En el panel principal se muestra el título “App Streamlit de Reducción de Dimensionalidad” acompañado de una breve descripción, y un mensaje informativo que indica al usuario que debe cargar un archivo CSV/XLSX desde la barra lateral para comenzar a trabajar con la aplicación.

Este estado refleja que la aplicación está lista para iniciar el procesamiento de datos, a la espera de que se cargue un dataset.
![003](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/003.png)

> ### 5.-
En esta imagen se puede observar la aplicación de Streamlit en funcionamiento, luego de haber cargado correctamente el archivo Dry_Bean_Dataset.xlsx de 3 MB.

En el panel lateral izquierdo, dentro de la sección 1) Carga de datos, aparece el archivo ya cargado. En la sección 2) Configuración de columnas, se mantiene activa la opción que indica que la primera fila corresponde a encabezados, lo cual permite interpretar correctamente las variables del dataset.

En el panel principal de la aplicación se despliega el título “App Streamlit de Reducción de Dimensionalidad”, seguido de la sección “Vista previa de datos”, donde se muestra una tabla con las primeras filas del dataset. Entre las columnas visibles se incluyen variables como Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent y Solidity.

Este estado refleja que el sistema ya está listo para iniciar el preprocesamiento y aplicar los métodos de reducción de dimensionalidad configurados en la aplicación.
![004](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/004.png)

> ### 6.-
En esta imagen se puede observar la sección “Vista previa de datos” de la aplicación en Streamlit, donde se muestran algunas de las filas del dataset previamente cargado.

La tabla despliega distintas variables numéricas asociadas a las características de las muestras. Entre ellas destacan:

* Area: superficie medida.
* Perimeter: perímetro de la figura.
* MajorAxisLength y MinorAxisLength: longitudes de los ejes mayor y menor.
* AspectRation: relación de aspecto entre los ejes.
* Eccentricity: grado de desviación respecto a una forma circular.
* ConvexArea: área convexa que encierra la figura.
* EquivDiameter: diámetro equivalente.
* Extent: proporción del área de la figura respecto al área del rectángulo que la contiene.
* Solidity: relación entre el área de la figura y el área convexa.

Este estado refleja que los datos han sido procesados correctamente por la aplicación y están listos para ser utilizados en los métodos de reducción de dimensionalidad (PCA, LDA, t-SNE y UMAP) que forman parte del flujo de análisis.
![005](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/005.png)

> ### 7.-
En esta imagen se puede observar la sección “Proyección 2D del embedding” generada por la aplicación de Streamlit tras aplicar el método de Análisis de Componentes Principales (PCA).

El gráfico de dispersión muestra los datos proyectados en dos dimensiones principales:

* El eje Emb1 corresponde al primer componente principal, que captura la mayor varianza posible de los datos originales.
* El eje Emb2 representa el segundo componente principal, ortogonal al primero y encargado de capturar la mayor varianza restante.

La distribución de los puntos revela la estructura interna del dataset, evidenciando concentraciones y posibles agrupamientos de observaciones. Esta representación facilita la interpretación visual de la información de alta dimensionalidad reducida a un plano bidimensional, permitiendo identificar patrones, tendencias y relaciones entre las muestras analizadas.
![006](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/006.png)

> ### 8.-
En esta imagen se puede observar la métrica no supervisada Silhouette, obtenida mediante la aplicación del algoritmo k-means sobre el embedding en dos dimensiones.

El valor reportado es de 0.495, lo que indica un nivel moderado de cohesión y separación entre los grupos formados. La métrica Silhouette varía entre -1 y 1:

* Valores cercanos a 1 reflejan que los clusters están bien definidos y separados.
* Valores cercanos a 0 sugieren solapamiento entre grupos.
* Valores negativos implican que los datos pueden estar mal asignados a los clusters.

En este caso, el resultado de 0.495 sugiere que los clusters generados presentan una estructura razonable, con una separación aceptable entre ellos, aunque no completamente óptima.
![007](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/007.png)

> ### 9.-
En esta imagen se puede observar la sección correspondiente a la métrica supervisada de Accuracy, calculada mediante el algoritmo k-Nearest Neighbors (kNN).

El mensaje mostrado indica que aún no se ha definido una columna target en la barra lateral de configuración. Esta columna es necesaria para evaluar el desempeño del modelo supervisado, ya que representa la variable de clase o etiqueta con la que se comparan las predicciones generadas por el algoritmo.

Hasta que no se seleccione un atributo como target, la aplicación no podrá calcular ni mostrar el valor de accuracy, lo que resalta la importancia de identificar correctamente la variable dependiente en el dataset para llevar a cabo la evaluación supervisada.

![008](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/008.png)

> ### 10.-

En esta imagen se puede observar la sección “Exportar resultados” de la aplicación desarrollada en Streamlit.

Se muestra disponible un botón con la opción “Descargar embedding (CSV)”, el cual permite al usuario exportar los resultados del proceso de reducción de dimensionalidad en formato CSV. Esta funcionalidad facilita la reutilización de los datos procesados, ya sea para análisis posteriores, integración con otras herramientas o almacenamiento para futuras consultas.

La presencia de esta opción confirma que la aplicación no solo permite la exploración interactiva de los embeddings, sino también su preservación y portabilidad en un formato ampliamente utilizado.

![009](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/009.png)

> ### 11.-

En esta imagen se puede observar la sección “Notas y recomendaciones” de la aplicación en Streamlit, presentada en un cuadro desplegable para consulta del usuario.

En esta sección se detallan consideraciones importantes sobre el uso de los distintos métodos y métricas:

* t-SNE es sensible a los parámetros perplexity y learning_rate, por lo que pequeños cambios pueden modificar la forma de los clusters.
* UMAP tiende a conservar mejor la estructura local con valores bajos de n_neighbors y la global con valores más altos.
* LDA requiere una columna target y puede generar como máximo n_clases − 1 dimensiones.
* PCA es un método lineal útil para capturar la varianza global y también puede emplearse como paso previo en otros análisis.
* Para el cálculo de accuracy, se aplica un particionado 80/20 utilizando el algoritmo kNN sobre el embedding 2D.
* Para la métrica de silhouette, se utiliza el algoritmo k-means aplicado igualmente sobre el embedding 2D.

Este apartado cumple la función de guía práctica, orientando al usuario en la interpretación y correcta aplicación de los métodos de reducción de dimensionalidad y de las métricas incluidas en la aplicación.

![010](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/010.png)

> ### 12.-

ChatGPT Plus

En esta imagen se puede observar la sección “1) Carga de datos” dentro de la aplicación de Streamlit.

El panel ofrece la posibilidad de subir un archivo en formato CSV o XLSX, ya sea arrastrándolo directamente al recuadro o seleccionándolo mediante el botón “Browse files”. Se indica un límite de carga de 200 MB por archivo.

Debajo del cuadro de carga aparece el archivo Dry_Bean_Dataset.xlsx, con un tamaño de 3.0 MB, confirmando que el dataset ha sido cargado correctamente en la aplicación y está listo para ser utilizado en las etapas posteriores de configuración y análisis.

![011](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/011.png)

> ### 13.-

En esta imagen se puede observar la sección “2) Configuración de columnas” dentro de la aplicación de Streamlit.

El panel permite definir distintos parámetros relacionados con la estructura del dataset:

* La opción “La primera fila tiene encabezados” aparece marcada, lo que indica que la aplicación reconocerá los nombres de las variables a partir de la primera fila del archivo cargado.
* El campo “Separador (solo CSV)” está configurado con la coma (,) como delimitador por defecto.
* Se incluye un espacio para especificar el Encoding, en caso de ser necesario, aunque puede dejarse vacío para que se detecte automáticamente.
* El apartado “Columna ID (opcional)” permite seleccionar una columna que identifique de manera única cada fila del dataset.
* Finalmente, el campo “Columna target (clase)” es esencial para los métodos supervisados, ya que define la variable objetivo necesaria para ejecutar algoritmos como LDA y calcular la métrica de accuracy con kNN.

Este módulo ofrece al usuario control sobre la correcta interpretación de los datos, asegurando un preprocesamiento adecuado para el análisis posterior.

![012](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/012.png)

![013](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/013.png)

![014](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/014.png)

![015](https://github.com/GuidoRiosCiaffaroni/Streamlit/blob/main/img/015.png)

