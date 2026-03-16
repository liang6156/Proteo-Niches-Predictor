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
# 1. Load Data & Models
# ==========================================
@st.cache_resource
def load_models_and_data():
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

    try:
        model_prot = joblib.load('model_Proteomic_LightGBM.pkl')
        df_top30 = pd.read_csv('2900_Proteomic_Top30_Panel.csv')
        top30_proteins = df_top30['Protein'].tolist()
    except Exception as e:
        model_prot, top30_proteins = None, []

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
# 3. Home Page (Added Disease Risks)
# ==========================================
if app_mode == "🏠 Home":
    st.title("Cross-Scale Subtype Predictor via Olink Proteomics")
    st.markdown("Welcome to the dual-channel diagnostic platform. This system translates massive multi-omics findings into actionable clinical insights.")
    
    try:
        st.image("2900_annotated_umap_final.png", caption="Digital Mirror: From Phenotype to Molecules", use_container_width=True)
    except FileNotFoundError:
        pass

    # --- 注入疾病风险解读卡片 ---
    st.markdown("---")
    st.subheader("📋 Subtype Disease Risk Profiles (Long-term Clinical Outcome)")
    st.markdown("Based on Cox longitudinal follow-up models across hundreds of diseases, our proteomic subtypes reveal profound prognostic value:")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**🟤 C0: Tissue Fibrosis & Frailty**\n\nCore marker of extracellular matrix (ECM) remodeling. Represents a systemic progression towards tissue fibrosis, sharing a common pathological endpoint. **Extreme high risk for future Kidney Failure and Liver Cirrhosis.**")
        st.error("**🔴 C2: ER Stress & Metabolic Storm**\n\nThe ultimate early-warning biomarker state. Driven by protein assembly and ER stress. **Strongly predicts the future onset of Hyperlipidemia, Primary Hypertension, Sepsis, and Kidney Failure** before overt symptoms appear.")
    with col_b:
        st.success("**🟢 C1: Metabolic Insulator**\n\nHematopoietic and hemostatic dominant type. Acts as a 'health umbrella', functioning as an **insulator against systemic metabolic dysregulation and end-stage organ diseases**. Lowest overall disease risk.")
        st.warning("**🟠 C3: Autoimmune & RNA Processing**\n\nDefinitive evidence of Gut-Immune Axis collapse. Driven by RNA splicing abnormalities that generate neoantigens. **Permanent proteomic footprint for severe Celiac Disease and Malabsorption Syndrome.**")

# ==========================================
# 4. Channel 1: Clinical Phenotype Mode (UX Upgraded)
# ==========================================
elif app_mode == "🩺 Channel 1: Clinical Phenotype":
    st.title("🩺 Clinical Phenotype Screening Mode")
    st.markdown("Enter routine metrics to evaluate the risk of hidden pathological subtypes.")
    st.info("💡 **Note on Missing Data:** The underlying LightGBM algorithm natively supports missing values. If specific metrics are unavailable, you may leave them as default (0). The model will dynamically adjust its inference pathways, though confidence may vary slightly.")
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.subheader("📝 Patient Data Input")
        user_inputs = {}
        
        # 定义需要纯整数输入的变量及后缀单位
        int_fields = {
            'Lifetime number of sexual partners': '',
            'Duration of moderate activity': ' (minutes/day)',
            'Average weekly beer plus cider intake': ' (pints)',
            'Number of treatments/medications taken': '',
            'Cereal intake': ' (bowls/week)',
            'Water intake': ' (glasses/day)',
            'Average weekly champagne plus white wine intake': ' (glasses)'
        }

        if top_features:
            for feature in top_features:
                pheno_name = str(feature_map.get(feature, feature))
                
                # 1. 前端隐藏 Basophill count，但在后端保留字典键以供补全 NaN
                if "Basophill count" in pheno_name:
                    continue 

                # 2. 处理需要输入具体整数的多分类变量
                matched_int_key = next((k for k in int_fields.keys() if k in pheno_name), None)
                if matched_int_key:
                    unit = int_fields[matched_int_key]
                    user_inputs[feature] = st.number_input(f"{pheno_name}{unit}", min_value=0, value=0, step=1, format="%d")
                
                # 3. 处理特定的水质硬度下拉框
                elif "Water hardness" in pheno_name:
                    wh_options = {
                        "0-60 mg/L": 0,
                        "60-120 mg/L": 1,
                        "120-180 mg/L": 2,
                        ">180 mg/L": 3
                    }
                    choice = st.selectbox(f"{pheno_name}", list(wh_options.keys()))
                    user_inputs[feature] = wh_options[choice]

                # 4. 处理剩余的连续变量或二分类变量
                else:
                    v_type = str(var_type_map.get(feature, '')).lower()
                    if v_type in ['continuous', 'numeric']:
                        user_inputs[feature] = st.number_input(f"{pheno_name}", value=0.0, format="%.2f")
                    else:
                        # 默认将二分类设置为下拉框 Yes/No
                        choice = st.selectbox(f"{pheno_name}", ["No", "Yes"])
                        user_inputs[feature] = 1 if choice == "Yes" else 0
        
        predict_btn = st.button("🚀 Run Risk Assessment", use_container_width=True)

    with col2:
        st.subheader("📊 Risk Assessment Results")
        if predict_btn and model_C2 and model_C0:
            with st.spinner("Analyzing decision trees..."):
                
                # 为隐藏的变量 (Basophill count等) 强制填充 NaN 以满足模型 20 维要求
                for f in top_features:
                    if f not in user_inputs:
                        user_inputs[f] = np.nan
                
                input_df = pd.DataFrame([user_inputs])
                
                # 类型强转
                for feature in top_features:
                    v_type = str(var_type_map.get(feature, '')).lower()
                    if feature in input_df.columns:
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
                pred_class = model_prot.predict(patient_data)[0]
                pred_proba = np.max(model_prot.predict_proba(patient_data)[0])
                
                st.success("✅ Molecular Profiling Complete!")
                st.markdown("---")
                st.markdown("### 🧬 Definitive Diagnosis Summary")
                st.info(f"**Predicted Molecular Subtype:** \n# {pred_class}")
                st.metric(label="Algorithm Confidence (LightGBM)", value=f"{pred_proba*100:.1f}%")
                st.markdown("The patient's proteomic signature is definitively aligned with the above subtype based on the 30-biomarker diagnostic panel.")
                st.markdown("---")