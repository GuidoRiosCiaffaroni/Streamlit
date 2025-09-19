### Texto de la clase

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
