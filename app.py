import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. Page Configuration (Global English)
# ==========================================
st.set_page_config(
    page_title="Olink Subtype Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 1. Load Data & Models (Cached for Performance)
# ==========================================
@st.cache_resource
def load_models_and_data():
    # --- Load Channel 1: Clinical Models ---
    try:
        model_C2 = joblib.load('model_C2_LightGBM.pkl')
        model_C0 = joblib.load('model_C0_LightGBM.pkl')
        
        df_top20 = pd.read_csv('2900_LightGBM_Top20_Features_List.csv')
        top_features = df_top20['Feature_Code'].tolist()
        feature_map = dict(zip(top_features, df_top20['Phenotype'].tolist()))
        
        df_dict = pd.read_csv('2900_summary_health_related_traits.csv').drop_duplicates(subset=['name'])
        var_type_map = dict(zip(df_dict['name'], df_dict['varType']))
    except Exception as e:
        model_C2, model_C0, top_features, feature_map, var_type_map = None, None, [], {}, {}
        st.error(f"Channel 1 Model Loading Error: {e}")

    # --- Load Channel 2: Proteomic Models ---
    try:
        # 只保留轻量级的 LightGBM 预测模型，移除 UMAP 加载
        model_prot = joblib.load('model_Proteomic_LightGBM.pkl')
        
        df_top30 = pd.read_csv('2900_Proteomic_Top30_Panel.csv')
        top30_proteins = df_top30['Protein'].tolist()
    except Exception as e:
        model_prot, top30_proteins = None, []
        st.error(f"Channel 2 Proteomic Model Loading Error: {e}")

    return model_C2, model_C0, top_features, feature_map, var_type_map, model_prot, top30_proteins

(model_C2, model_C0, top_features, feature_map, 
 var_type_map, model_prot, top30_proteins) = load_models_and_data()

# ==========================================
# 2. Sidebar Navigation
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3024/3024310.png", width=80) 
st.sidebar.title("🧬 Navigation")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["🏠 Home", 
     "🩺 Channel 1: Clinical Phenotype", 
     "🧪 Channel 2: Proteomic Precision"]
)
st.sidebar.markdown("---")
st.sidebar.info("Developed using UK Biobank multi-omics cohort. For research purposes only.")

# ==========================================
# 3. Home Page
# ==========================================
if app_mode == "🏠 Home":
    st.title("Cross-Scale Subtype Predictor via Olink Proteomics")
    st.markdown("""
    Welcome to the dual-channel diagnostic platform. This system translates massive multi-omics findings into actionable clinical insights.
    
    **Dual-Channel Prediction Engine:**
    * **Channel 1 (Clinical Phenotype)**: Rapid screening for hidden risks of **C2 (Metabolic Storm)** or **C0 (Frailty)** using 20 routine health metrics.
    * **Channel 2 (Proteomic Precision)**: High-precision patient stratification. Input 30 core Olink proteins to map patients directly into their exact molecular subtype (AUC > 0.97).
    """)
    try:
        st.image("2900_annotated_umap_final.png", caption="Digital Mirror: From Phenotype to Molecules", use_container_width=True)
    except FileNotFoundError:
        pass

# ==========================================
# 4. Channel 1: Clinical Phenotype Mode
# ==========================================
elif app_mode == "🩺 Channel 1: Clinical Phenotype":
    st.title("🩺 Clinical Phenotype Screening Mode")
    st.markdown("Enter routine metrics to evaluate the risk of hidden pathological subtypes.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📝 Patient Data Input")
        user_inputs = {}
        if top_features:
            for feature in top_features:
                pheno_name = feature_map.get(feature, feature)
                v_type = str(var_type_map.get(feature, '')).lower()
                if v_type in ['continuous', 'integer', 'numeric']:
                    user_inputs[feature] = st.number_input(f"{pheno_name}", value=0.0)
                else:
                    user_inputs[feature] = st.text_input(f"{pheno_name} (Code)", value="1")
        
        predict_btn = st.button("🚀 Run Risk Assessment", use_container_width=True)

    with col2:
        st.subheader("📊 Risk Assessment Results")
        if predict_btn and model_C2 and model_C0:
            with st.spinner("Analyzing decision trees..."):
                input_df = pd.DataFrame([user_inputs])
                for feature in top_features:
                    v_type = str(var_type_map.get(feature, '')).lower()
                    if v_type in ['continuous', 'integer', 'numeric']:
                        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce')
                    else:
                        input_df[feature] = input_df[feature].astype('category')
                try:
                    c2_risk = model_C2.predict_proba(input_df)[0][1]
                    c0_risk = model_C0.predict_proba(input_df)[0][1]
                    
                    st.markdown(f"#### 🔴 C2 Subtype (Metabolic/Inflammation): **{c2_risk*100:.1f}%**")
                    st.progress(float(c2_risk))
                    if c2_risk > 0.4: st.error("🚨 **HIGH RISK**: C2 characteristics detected.")
                    else: st.success("✅ C2 Risk is low.")
                        
                    st.markdown("---")
                    st.markdown(f"#### 🟤 C0 Subtype (Organ Frailty/Fibrosis): **{c0_risk*100:.1f}%**")
                    st.progress(float(c0_risk))
                    if c0_risk > 0.4: st.warning("⚠️ **ELEVATED RISK**: C0 frailty pattern detected.")
                    else: st.success("✅ C0 Risk is low.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# ==========================================
# 5. Channel 2: Proteomic Precision
# ==========================================
elif app_mode == "🧪 Channel 2: Proteomic Precision":
    st.title("🧪 Precision Proteomic Diagnostic")
    st.markdown("Upload a CSV containing the patient's core Olink protein expressions. The system will run extreme high-precision molecular inference.")
    
    if top30_proteins:
        with st.expander("📌 View Required Core Proteins Panel (Top 30)", expanded=False):
            st.code(", ".join(top30_proteins))
        
        template_df = pd.DataFrame(columns=top30_proteins)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Input CSV Template",
            data=csv_template,
            file_name="Patient_Proteomic_Template.csv",
            mime="text/csv",
        )
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Patient Proteomic Data (.csv)", type=["csv"])
    
    if uploaded_file is not None and model_prot:
        patient_df = pd.read_csv(uploaded_file)
        
        missing_cols = [col for col in top30_proteins if col not in patient_df.columns]
        if missing_cols:
            st.error(f"❌ Uploaded CSV is missing the following required proteins: {missing_cols}")
        else:
            with st.spinner("Running Proteomic Inference..."):
                patient_data = patient_df[top30_proteins].head(1).apply(pd.to_numeric, errors='coerce').fillna(0)
                
                # 仅执行快速精确的模型推理
                pred_class = model_prot.predict(patient_data)[0]
                pred_proba = np.max(model_prot.predict_proba(patient_data)[0])
                
                st.success("✅ Molecular Profiling Complete!")
                
                # 直接居中展示核心诊断结果，剥离视觉干扰
                st.markdown("---")
                st.markdown("### 🧬 Definitive Diagnosis Summary")
                st.info(f"**Predicted Molecular Subtype:** \n# {pred_class}")
                st.metric(label="Algorithm Confidence (LightGBM)", value=f"{pred_proba*100:.1f}%")
                st.markdown("The patient's proteomic signature is definitively aligned with the above subtype based on the 30-biomarker diagnostic panel.")
                st.markdown("---")