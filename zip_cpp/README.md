# dimred-ccp

App de Streamlit para reducción de dimensionalidad empaquetada como Common Code Package (CCP).

## Instalación (dev)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .[umap,export]
```

## Ejecución
```bash
dimred-serve
```

## Docker
```bash
docker build -t dimred-ccp:0.1.0 .
docker run --rm -p 8501:8501 dimred-ccp:0.1.0
```
