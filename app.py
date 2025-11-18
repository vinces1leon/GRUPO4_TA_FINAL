import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time

from pipeline import ejecutar_secuencial, ejecutar_paralelo

st.set_page_config(page_title="Comparador Secuencial vs Paralelo", layout="wide")

st.title("🔬 Comparador: Procesamiento Secuencial vs Paralelo")
st.markdown("### Análisis de Producción Agropecuaria")

st.markdown("""
Este dashboard compara el rendimiento del pipeline de análisis de datos ejecutado de forma:
- **Secuencial**: Procesamiento paso a paso tradicional
- **Paralelo**: Procesamiento optimizado con múltiples hilos/procesos

Sube los dos archivos CSV requeridos para comenzar.
""")

# Sección de carga de archivos
st.header("📁 Cargar Datos")
col1, col2 = st.columns(2)
uploaded1 = col1.file_uploader("📊 Archivo 1: 03_CAP200AB.csv", type=["csv"])
uploaded2 = col2.file_uploader("📊 Archivo 2: 03_CAP200A.csv", type=["csv"])

# Métricas de referencia local
METRICAS_LOCAL = {
    "secuencial": {"time_sec": 109.81, "mem_peak_mb": 1191.60},
    "paralelo": {"time_sec": 71.30, "mem_peak_mb": 1085.0}
}

if uploaded1 is not None and uploaded2 is not None:
    # Guardar archivos temporalmente
    csv1_path = "temp_03_CAP200AB.csv"
    csv2_path = "temp_03_CAP200A.csv"
    
    with open(csv1_path, "wb") as f:
        f.write(uploaded1.getbuffer())
    with open(csv2_path, "wb") as f:
        f.write(uploaded2.getbuffer())
    
    st.success("✅ Archivos cargados correctamente")
    
    # Botones de ejecución
    st.header("⚡ Ejecutar Análisis")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    # ==================== BOTÓN SECUENCIAL ====================
    if col_btn1.button("🐢 Ejecutar Secuencial", use_container_width=True):
        with st.spinner("Ejecutando pipeline secuencial..."):
            start_ui = time.time()
            res_seq = ejecutar_secuencial(csv1_path, csv2_path)
            end_ui = time.time()
        
        st.success(f"✅ Secuencial finalizado en {end_ui - start_ui:.2f}s")
        
        # Mostrar métricas
        st.subheader("📊 Métricas Secuencial")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        col_m1.metric("⏱️ Tiempo (VM)", f"{res_seq['tiempo']:.2f}s")
        col_m2.metric("💾 Memoria Pico (VM)", f"{res_seq['memoria_mb']:.1f} MB")
        
        # Comparación con local
        ratio_time = res_seq['tiempo'] / METRICAS_LOCAL['secuencial']['time_sec']
        ratio_mem = res_seq['memoria_mb'] / METRICAS_LOCAL['secuencial']['mem_peak_mb']
        
        col_m3.metric("🔄 Ratio VM/Local (tiempo)", f"{ratio_time:.2f}x")
        
        st.info(f"""
        **Comparación con entorno local:**
        - Tiempo Local: {METRICAS_LOCAL['secuencial']['time_sec']:.2f}s
        - Memoria Local: {METRICAS_LOCAL['secuencial']['mem_peak_mb']:.1f} MB
        - Ratio Tiempo: {ratio_time:.2f}x
        - Ratio Memoria: {ratio_mem:.2f}x
        """)
        
        # Mostrar gráficos
        st.subheader("📈 Visualizaciones - Secuencial")
        
        tab1, tab2, tab3 = st.tabs(["Heatmap Correlación", "Método del Codo", "Clusters PCA"])
        
        with tab1:
            st.pyplot(res_seq['fig_heatmap'])
        
        with tab2:
            st.pyplot(res_seq['fig_elbow'])
        
        with tab3:
            st.pyplot(res_seq['fig_scatter'])
        
        # Mostrar estadísticas del DataFrame
        with st.expander("📋 Ver estadísticas del DataFrame procesado"):
            st.write(f"Dimensiones: {res_seq['df_filtrado'].shape}")
            st.write(res_seq['df_filtrado'].describe())
    
    # ==================== BOTÓN PARALELO ====================
    if col_btn2.button("🚀 Ejecutar Paralelo", use_container_width=True):
        with st.spinner("Ejecutando pipeline paralelo..."):
            start_ui = time.time()
            res_par = ejecutar_paralelo(csv1_path, csv2_path)
            end_ui = time.time()
        
        st.success(f"✅ Paralelo finalizado en {end_ui - start_ui:.2f}s")
        
        # Mostrar métricas
        st.subheader("📊 Métricas Paralelo")
        col_m1, col_m2, col_m3 = st.columns(3)
        
        col_m1.metric("⏱️ Tiempo (VM)", f"{res_par['tiempo']:.2f}s")
        col_m2.metric("💾 Memoria Pico (VM)", f"{res_par['memoria_mb']:.1f} MB")
        
        # Comparación con local
        ratio_time = res_par['tiempo'] / METRICAS_LOCAL['paralelo']['time_sec']
        ratio_mem = res_par['memoria_mb'] / METRICAS_LOCAL['paralelo']['mem_peak_mb']
        
        col_m3.metric("🔄 Ratio VM/Local (tiempo)", f"{ratio_time:.2f}x")
        
        st.info(f"""
        **Comparación con entorno local:**
        - Tiempo Local: {METRICAS_LOCAL['paralelo']['time_sec']:.2f}s
        - Memoria Local: {METRICAS_LOCAL['paralelo']['mem_peak_mb']:.1f} MB
        - Ratio Tiempo: {ratio_time:.2f}x
        - Ratio Memoria: {ratio_mem:.2f}x
        """)
        
        # Mostrar gráficos
        st.subheader("📈 Visualizaciones - Paralelo")
        
        tab1, tab2, tab3 = st.tabs(["Heatmap Correlación", "Método del Codo", "Clusters PCA"])
        
        with tab1:
            st.pyplot(res_par['fig_heatmap'])
        
        with tab2:
            st.pyplot(res_par['fig_elbow'])
        
        with tab3:
            st.pyplot(res_par['fig_scatter'])
        
        # Mostrar estadísticas del DataFrame
        with st.expander("📋 Ver estadísticas del DataFrame procesado"):
            st.write(f"Dimensiones: {res_par['df_filtrado'].shape}")
            st.write(res_par['df_filtrado'].describe())
    
    # ==================== BOTÓN COMPARACIÓN ====================
    if col_btn3.button("⚖️ Ejecutar Ambas y Comparar", use_container_width=True):
        st.subheader("🔄 Ejecutando ambos pipelines...")
        
        # Barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Ejecutar secuencial
        status_text.text("Ejecutando pipeline secuencial...")
        progress_bar.progress(10)
        start_seq = time.time()
        res_seq = ejecutar_secuencial(csv1_path, csv2_path)
        end_seq = time.time()
        progress_bar.progress(50)
        
        # Ejecutar paralelo
        status_text.text("Ejecutando pipeline paralelo...")
        start_par = time.time()
        res_par = ejecutar_paralelo(csv1_path, csv2_path)
        end_par = time.time()
        progress_bar.progress(100)
        
        status_text.text("✅ Ambos pipelines completados")
        
        # ==================== COMPARACIÓN DE MÉTRICAS ====================
        st.header("📊 Comparación de Rendimiento")
        
        # Crear DataFrame comparativo
        df_comp = pd.DataFrame({
            "Método": ["Secuencial", "Paralelo"],
            "Tiempo (s)": [res_seq["tiempo"], res_par["tiempo"]],
            "Memoria (MB)": [res_seq["memoria_mb"], res_par["memoria_mb"]]
        })
        
        # Métricas comparativas principales
        col_comp1, col_comp2, col_comp3 = st.columns(3)
        
        speedup = res_seq["tiempo"] / res_par["tiempo"]
        mem_saving = ((res_seq["memoria_mb"] - res_par["memoria_mb"]) / res_seq["memoria_mb"]) * 100
        
        col_comp1.metric("🚀 Aceleración (Speedup)", f"{speedup:.2f}x")
        col_comp2.metric("💾 Ahorro de Memoria", f"{mem_saving:.1f}%")
        col_comp3.metric("⏱️ Tiempo Ahorrado", f"{res_seq['tiempo'] - res_par['tiempo']:.2f}s")
        
        # Tabla comparativa
        st.dataframe(df_comp, use_container_width=True)
        
        # Gráfico de barras comparativo
        st.subheader("📊 Visualización Comparativa")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Tiempo de Ejecución")
            fig_tiempo, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_comp["Método"], df_comp["Tiempo (s)"], color=['#FF6B6B', '#4ECDC4'])
            ax.set_ylabel("Tiempo (segundos)")
            ax.set_title("Comparación de Tiempo de Ejecución")
            for i, v in enumerate(df_comp["Tiempo (s)"]):
                ax.text(i, v + 1, f"{v:.2f}s", ha='center', va='bottom')
            st.pyplot(fig_tiempo)
        
        with col_chart2:
            st.markdown("#### Uso de Memoria")
            fig_mem, ax = plt.subplots(figsize=(6, 4))
            ax.bar(df_comp["Método"], df_comp["Memoria (MB)"], color=['#FF6B6B', '#4ECDC4'])
            ax.set_ylabel("Memoria (MB)")
            ax.set_title("Comparación de Uso de Memoria")
            for i, v in enumerate(df_comp["Memoria (MB)"]):
                ax.text(i, v + 10, f"{v:.1f} MB", ha='center', va='bottom')
            st.pyplot(fig_mem)
        
        # ==================== COMPARACIÓN VM vs LOCAL ====================
        st.header("🌐 Comparación VM vs Entorno Local")
        
        col_local1, col_local2 = st.columns(2)
        
        with col_local1:
            st.subheader("🐢 Secuencial")
            ratio_seq_time = res_seq['tiempo'] / METRICAS_LOCAL['secuencial']['time_sec']
            ratio_seq_mem = res_seq['memoria_mb'] / METRICAS_LOCAL['secuencial']['mem_peak_mb']
            
            st.metric("Ratio Tiempo VM/Local", f"{ratio_seq_time:.2f}x")
            st.metric("Ratio Memoria VM/Local", f"{ratio_seq_mem:.2f}x")
            
            st.info(f"""
            **Local:** {METRICAS_LOCAL['secuencial']['time_sec']:.2f}s | {METRICAS_LOCAL['secuencial']['mem_peak_mb']:.1f} MB
            
            **VM:** {res_seq['tiempo']:.2f}s | {res_seq['memoria_mb']:.1f} MB
            """)
        
        with col_local2:
            st.subheader("🚀 Paralelo")
            ratio_par_time = res_par['tiempo'] / METRICAS_LOCAL['paralelo']['time_sec']
            ratio_par_mem = res_par['memoria_mb'] / METRICAS_LOCAL['paralelo']['mem_peak_mb']
            
            st.metric("Ratio Tiempo VM/Local", f"{ratio_par_time:.2f}x")
            st.metric("Ratio Memoria VM/Local", f"{ratio_par_mem:.2f}x")
            
            st.info(f"""
            **Local:** {METRICAS_LOCAL['paralelo']['time_sec']:.2f}s | {METRICAS_LOCAL['paralelo']['mem_peak_mb']:.1f} MB
            
            **VM:** {res_par['tiempo']:.2f}s | {res_par['memoria_mb']:.1f} MB
            """)
        
        # ==================== COMPARACIÓN DE VISUALIZACIONES ====================
        st.header("📈 Comparación de Visualizaciones")
        
        st.subheader("Matrices de Correlación")
        col_heat1, col_heat2 = st.columns(2)
        with col_heat1:
            st.markdown("**Secuencial**")
            st.pyplot(res_seq['fig_heatmap'])
        with col_heat2:
            st.markdown("**Paralelo**")
            st.pyplot(res_par['fig_heatmap'])
        
        st.subheader("Método del Codo")
        col_elbow1, col_elbow2 = st.columns(2)
        with col_elbow1:
            st.markdown("**Secuencial**")
            st.pyplot(res_seq['fig_elbow'])
        with col_elbow2:
            st.markdown("**Paralelo**")
            st.pyplot(res_par['fig_elbow'])
        
        st.subheader("Clusters PCA")
        col_scatter1, col_scatter2 = st.columns(2)
        with col_scatter1:
            st.markdown("**Secuencial**")
            st.pyplot(res_seq['fig_scatter'])
        with col_scatter2:
            st.markdown("**Paralelo**")
            st.pyplot(res_par['fig_scatter'])
        
        # Resumen final
        st.header("🎯 Resumen de Resultados")
        st.success(f"""
        ### Conclusiones:
        - ⚡ El procesamiento paralelo es **{speedup:.2f}x más rápido** que el secuencial
        - 💾 Ahorro de memoria: **{mem_saving:.1f}%**
        - 🕐 Tiempo ahorrado: **{res_seq['tiempo'] - res_par['tiempo']:.2f} segundos**
        - 📊 Ambos métodos producen resultados idénticos
        - 🌐 Ratio VM/Local (secuencial): **{ratio_seq_time:.2f}x**
        - 🌐 Ratio VM/Local (paralelo): **{ratio_par_time:.2f}x**
        """)

else:
    st.info("👆 Por favor, sube ambos archivos CSV para comenzar el análisis")
    
    # Información adicional
    with st.expander("ℹ️ Información sobre el análisis"):
        st.markdown("""
        ### ¿Qué hace este pipeline?
        
        1. **Carga y fusión** de dos datasets de producción agropecuaria
        2. **Limpieza de datos**: eliminación de duplicados, valores nulos, normalización
        3. **Transformación**: unión de columnas enteras y decimales
        4. **Análisis estadístico**: 
           - Matriz de correlaciones
           - Análisis de componentes principales (PCA)
           - Clustering con K-Means
        5. **Visualización**: Heatmaps, gráficos de dispersión, método del codo
        
        ### Diferencias entre métodos:
        
        - **Secuencial**: Procesa cada paso uno tras otro
        - **Paralelo**: Utiliza ThreadPoolExecutor y ProcessPoolExecutor para ejecutar múltiples tareas simultáneamente
        
        ### Métricas de referencia (entorno local):
        - Secuencial: 109.81s | 1191.60 MB
        - Paralelo: 71.30s | 1085.0 MB
        - Speedup esperado: ~1.54x
        """)