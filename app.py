import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pipeline import ejecutar_secuencial, ejecutar_paralelo

st.set_page_config(page_title="Comparador Secuencial vs Paralelo", layout="wide")
st.title("Comparador: versión secuencial vs versión paralela")

st.markdown("Sube los dos CSVs requeridos (03_CAP200AB.csv y 03_CAP200A.csv).")

col1, col2 = st.columns(2)
uploaded1 = col1.file_uploader("CSV 1 (03_CAP200AB.csv)", type=["csv"])
uploaded2 = col2.file_uploader("CSV 2 (03_CAP200A.csv)", type=["csv"])

# métricas locales (tus valores de referencia)
METRICAS_LOCAL = {
    "secuencial": {"time_sec": 109.81, "mem_peak_mb": 1191.60},
    "paralelo": {"time_sec": 71.30, "mem_peak_mb": 1085.0}
}

if uploaded1 is not None and uploaded2 is not None:
    csv1_path = "uploaded_03_CAP200AB.csv"
    csv2_path = "uploaded_03_CAP200A.csv"
    with open(csv1_path, "wb") as f:
        f.write(uploaded1.getbuffer())
    with open(csv2_path, "wb") as f:
        f.write(uploaded2.getbuffer())

    st.success("Archivos cargados. Puedes ejecutar las pruebas abajo.")

    col_exec1, col_exec2, col_exec3 = st.columns(3)
    if col_exec1.button("Ejecutar secuencial"):
        with st.spinner("Ejecutando secuencial..."):
            res = ejecutar_secuencial(csv1_path, csv2_path)
        st.success("Secuencial finalizado")
        st.metric("Tiempo (s) secuencial VM", f"{res['tiempo']:.2f}")
        st.metric("Memoria pico (MB) secuencial VM", f"{res['memoria_mb']:.1f}")
        # comparación vs local
        ratio = res['tiempo'] / METRICAS_LOCAL['secuencial']['time_sec']
        st.write(f"VM / Local (time_ratio) secuencial: **{ratio:.2f}x**")

        # Mostrar heatmap
        corr = res.get("correlacion")
        if corr is not None and not corr.empty:
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Heatmap - Secuencial")
            st.pyplot(fig)

        # Mostrar PCA scatter si existe
        X_pca = res.get("X_pca")
        dff = res.get("df_filtrado")
        if X_pca is not None and X_pca.size and dff is not None and 'Cluster' in dff.columns:
            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.scatter(X_pca[:,0], X_pca[:,1], c=dff['Cluster'], s=50, alpha=0.7)
            ax2.set_title("PCA Scatter - Secuencial")
            st.pyplot(fig2)

    if col_exec2.button("Ejecutar paralelo"):
        with st.spinner("Ejecutando paralelo..."):
            res = ejecutar_paralelo(csv1_path, csv2_path)
        st.success("Paralelo finalizado")
        st.metric("Tiempo (s) paralelo VM", f"{res['tiempo']:.2f}")
        st.metric("Memoria pico (MB) paralelo VM", f"{res['memoria_mb']:.1f}")
        # comparación vs local
        ratio_p = res['tiempo'] / METRICAS_LOCAL['paralelo']['time_sec']
        st.write(f"VM / Local (time_ratio) paralelo: **{ratio_p:.2f}x**")

        corr = res.get("correlacion")
        if corr is not None and not corr.empty:
            fig, ax = plt.subplots(figsize=(12,8))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
            ax.set_title("Heatmap - Paralelo")
            st.pyplot(fig)

        X_pca = res.get("X_pca")
        dfp = res.get("df_filtrado")
        if X_pca is not None and X_pca.size and dfp is not None and 'Cluster' in dfp.columns:
            fig2, ax2 = plt.subplots(figsize=(8,6))
            ax2.scatter(X_pca[:,0], X_pca[:,1], c=dfp['Cluster'], s=50, alpha=0.7)
            ax2.set_title("PCA Scatter - Paralelo")
            st.pyplot(fig2)

    if col_exec3.button("Ejecutar ambas y comparar"):
        with st.spinner("Ejecutando secuencial..."):
            res_seq = ejecutar_secuencial(csv1_path, csv2_path)
        with st.spinner("Ejecutando paralelo..."):
            res_par = ejecutar_paralelo(csv1_path, csv2_path)

        st.header("Comparación de métricas")
        df_comp = pd.DataFrame({
            "tipo": ["secuencial", "paralelo"],
            "tiempo_s": [res_seq["tiempo"], res_par["tiempo"]],
            "memoria_mb": [res_seq["memoria_mb"], res_par["memoria_mb"]]
        })
        st.dataframe(df_comp)

        # Mostrar heatmaps lado a lado
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Heatmap - Secuencial")
            if res_seq.get("correlacion") is not None:
                fig, ax = plt.subplots(figsize=(10,8))
                sns.heatmap(res_seq["correlacion"], annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)
        with c2:
            st.subheader("Heatmap - Paralelo")
            if res_par.get("correlacion") is not None:
                fig, ax = plt.subplots(figsize=(10,8))
                sns.heatmap(res_par["correlacion"], annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
                st.pyplot(fig)

else:
    st.info("Sube ambos archivos CSV para comenzar.")
