import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import streamlit.components.v1 as components
import io
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# --- Streamlit App Configuration and Blue Theme ---
st.set_page_config(
    page_title="Breast Cancer Diagnosis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# New color theme for better readability
main_bg_color = "#183e65"
secondary_bg_color = "#203c67"
text_color = "#FBF9F9"

st.markdown(f"""
<style>
/* Main app styling */
.stApp {{
    background-color: {main_bg_color};
    color: {text_color};
}}
/* General text and headings */
h1, h2, h3, h4, h5, h6, p, label, .st-bb {{
    color: {text_color};
}}
/* Text inside info boxes and other containers */
.st-emotion-cache-6t1mwy, .st-emotion-cache-12k0e1w, .st-emotion-cache-1c1i9v0 p {{
    color: {text_color};
}}
/* Fix for st.metric labels and values */
[data-testid="stMetricLabel"] p{{
    color: #F !important;
}}
[data-testid="stMetricValue"] h1 {{
    color: #4CAF50
}}
/* Sidebar colors */
.st-emotion-cache-1c7y3k9 {{
    background-color: {main_bg_color};
    color: {text_color};
}}
/* Tabs styling */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {{
    font-size: 1.2rem;
    color: {text_color};
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 20px;
}}
.st-emotion-cache-1r651z4, .st-emotion-cache-1r44j32, .st-emotion-cache-1b29s41 {{
    background-color: {secondary_bg_color};
    color: {text_color};
}}
.st-emotion-cache-1r651z4:hover, .st-emotion-cache-1r44j32:hover, .st-emotion-cache-1b29s41:hover {{
    background-color: #295b94;
}}

/* === UPDATED BUTTON STYLING === */
.stButton > button {{
    background-color: #e63946; /* Red background */
    color: white; /* White text */
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
}}
.stButton > button:hover {{
    background-color: #d12e3b; /* Darker red on hover */
}}
/* ============================== */

</style>
""", unsafe_allow_html=True)

# --- Title and Introduction ---
st.title("🔬 Breast Cancer Diagnosis using ML & Explainable AI")
st.markdown("""
Welcome to the interactive dashboard for breast cancer diagnosis. This application showcases various machine learning models and utilizes Explainable AI (XAI) techniques to provide deep insights into their predictions.
""")
st.markdown("---")

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads and preprocesses the breast cancer dataset."""
    df = pd.read_csv("data.csv")
    df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

@st.cache_resource
def train_and_evaluate_models(x_train, x_test, y_train, y_test):
    """Trains and evaluates multiple machine learning models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred) * 100
        results[name] = {
            "model": model,
            "accuracy": acc,
            "y_pred": y_pred
        }
    return results

data = load_data()
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']
feature_names = X.columns
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42, stratify=y)
results = train_and_evaluate_models(x_train, x_test, y_train, y_test)

# --- NEW Caching function for XAI results ---
@st.cache_resource
def get_cached_xai_results(model_choice, _selected_model, x_test_subset, _feature_names):
    """
    Computes and caches SHAP values and LIME explainers for a given model.
    The `model_choice` string is used as the hashable key.
    """
    with st.spinner(f"Computing SHAP values for {model_choice}... This may take a moment."):
        shap_explainer = shap.Explainer(_selected_model.predict_proba, x_test_subset, feature_names=_feature_names)
        shap_values = shap_explainer(x_test_subset)

    lime_explainer = LimeTabularExplainer(
        x_test_subset,
        mode='classification',
        feature_names=_feature_names,
        class_names=['Benign', 'Malignant']
    )
    return shap_values, lime_explainer

# --- Tabs for navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Performance", "🔍 Explainable AI", "📈 Feature Importance & Dependence", "❓ Ask the AI", "📝 Report-Based Prediction"])

with tab1:
    st.header("📊 Model Performance Comparison")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Accuracies")
        st.markdown("---")
        for name, res in results.items():
            st.metric(label=f"**{name}**", value=f"{res['accuracy']:.2f}%")
        
        st.subheader("Classification Reports")
        model_report = st.selectbox("Select model to view report:", list(results.keys()))
        report = classification_report(y_test, results[model_report]['y_pred'])
        st.text(report)

    with col2:
        st.subheader("Accuracy Bar Chart")
        fig_acc, ax_acc = plt.subplots(figsize=(8, 5), facecolor=main_bg_color)
        ax_acc.bar(results.keys(), [results[m]['accuracy'] for m in results], color=sns.color_palette("cividis", len(results)))
        ax_acc.set_ylabel("Accuracy (%)", color=text_color)
        ax_acc.tick_params(axis='x', colors=text_color)
        ax_acc.tick_params(axis='y', colors=text_color)
        ax_acc.set_title("Model Accuracy Comparison", color=text_color)
        ax_acc.grid(True, axis='y', linestyle='--', alpha=0.6, color='gray')
        ax_acc.set_facecolor(main_bg_color)
        st.pyplot(fig_acc)

with tab2:
    st.header("🔍 Explainable AI (XAI)")
    model_choice = st.selectbox("Select a model for XAI explanation:", list(results.keys()))
    
    # NEW: Defer the heavy computation until the button is clicked
    if st.button("Generate Explanations"):
        selected_model = results[model_choice]["model"]
        x_test_subset = x_test[:100]

        shap_values, lime_explainer = get_cached_xai_results(model_choice, selected_model, x_test_subset, feature_names)

        st.success(f"Explanations for {model_choice} have been generated. Switching back to this model will now be instant!")
        
        st.subheader("SHAP Beeswarm Plot (Global Explanation)")
        fig_bee, ax_bee = plt.subplots(figsize=(12, 6), facecolor=main_bg_color)
        shap.plots.beeswarm(shap_values[..., 1], max_display=10, show=False)
        ax_bee.tick_params(colors=text_color)
        plt.setp(ax_bee.get_yticklabels(), color=text_color)
        plt.setp(ax_bee.get_xticklabels(), color=text_color)
        ax_bee.set_facecolor(main_bg_color)
        st.pyplot(fig_bee)

        st.subheader("SHAP Summary Bar Plot (Global Explanation)")
        fig_bar, ax_bar = plt.subplots(figsize=(10, 6), facecolor=main_bg_color)
        shap.plots.bar(shap_values[..., 1], show=False)
        st.pyplot(fig_bar)

        st.subheader("SHAP Force Plot (Local Explanation)")
        st.info("The Force plot explains a single prediction.")
        
        instance_index = st.slider("Select a test instance to explain:", 0, len(x_test) - 1, 0)
        st.write(f"Explanation for test instance {instance_index}:")
        st.write(f"True Diagnosis: {'Malignant' if y_test.iloc[instance_index] == 1 else 'Benign'}")
        
        with st.spinner("Generating Force Plot..."):
            try:
                shap_instance_values = shap_values[instance_index, ..., 1]
                force_plot = shap.force_plot(
                    shap_instance_values.base_values, 
                    shap_instance_values.values,
                    shap_instance_values.data,
                    matplotlib=False
                )
                with io.StringIO() as output:
                    shap.save_html(output, force_plot)
                    shap_html = output.getvalue()
                components.html(shap_html, height=400)
            except Exception as e:
                st.error(f"Error generating Force Plot: {e}. If the error persists, please ensure your SHAP library is updated.")
                st.stop()


        st.subheader(f"LIME Explanation for {model_choice}")
        
        with st.spinner("Computing LIME explanation..."):
            exp = lime_explainer.explain_instance(
                x_test[instance_index], 
                selected_model.predict_proba, 
                num_features=5, 
                top_labels=1
            )
            components.html(exp.as_html(), height=400)
        st.markdown("---")

    else:
        st.info("Please select a model and click 'Generate Explanations' to view the XAI plots.")


with tab3:
    st.header("📈 Feature Importance & Dependence")
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("Permutation Importance")
        st.info("Permutation Importance measures the decrease in a model's score when a single feature is randomly shuffled, indicating its importance.")
        
        pi_model_choice = st.selectbox("Select model for Permutation Importance:", list(results.keys()), key="pi_model")
        model_pi = results[pi_model_choice]["model"]
        
        with st.spinner(f"Computing Permutation Importance for {pi_model_choice}..."):
            perm_result = permutation_importance(model_pi, x_test, y_test, n_repeats=10, random_state=42)
        sorted_idx = perm_result.importances_mean.argsort()

        fig_pi, ax_pi = plt.subplots(figsize=(10, 7), facecolor=main_bg_color)
        ax_pi.barh(np.array(feature_names)[sorted_idx], perm_result.importances_mean[sorted_idx], color=sns.color_palette("Set2", len(feature_names)))
        ax_pi.set_title(f"Permutation Importance for {pi_model_choice}", color=text_color)
        ax_pi.set_xlabel("Mean Importance", color=text_color)
        ax_pi.tick_params(axis='x', colors=text_color)
        ax_pi.tick_params(axis='y', colors=text_color)
        ax_pi.set_facecolor(main_bg_color)
        st.pyplot(fig_pi)

    with col4:
        st.subheader("Partial Dependence Plots")
        st.info("PDPs show the marginal effect of a single feature on the predicted outcome of a machine learning model.")
        
        pdp_model_choice = st.selectbox("Select model for PDP:", ["Logistic Regression"], key="pdp_model")
        features_to_plot = ['radius_mean', 'texture_mean', 'perimeter_mean']
        
        fig_pdp, ax_pdp = plt.subplots(figsize=(12, 6), ncols=3, sharey=True, facecolor=main_bg_color)
        PartialDependenceDisplay.from_estimator(
            results[pdp_model_choice]["model"],
            x_train,
            features=[feature_names.get_loc(f) for f in features_to_plot],
            feature_names=feature_names,
            ax=ax_pdp,
            line_kw={'color': '#f0f2f6'}
        )
        for ax in ax_pdp:
            ax.set_facecolor(main_bg_color)
            ax.tick_params(axis='x', colors=text_color)
            ax.tick_params(axis='y', colors=text_color)
            ax.set_ylabel("Partial dependence", color=text_color)
            ax.set_xlabel(ax.get_xlabel(), color=text_color)
        fig_pdp.suptitle(f"Partial Dependence Plots ({pdp_model_choice})", fontsize=16, color=text_color)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig_pdp)

    st.markdown("---")
    st.header("📌 Feature Correlation Heatmap")
    st.info("A heatmap to visualize the correlation matrix between all features in the dataset. Red indicates positive correlation, blue indicates negative.")
    fig_corr, ax_corr = plt.subplots(figsize=(14, 12), facecolor=main_bg_color)
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Correlation Matrix of All Features", color=text_color)
    ax_corr.tick_params(axis='x', colors=text_color)
    ax_corr.tick_params(axis='y', colors=text_color)
    ax_corr.set_facecolor(main_bg_color)
    st.pyplot(fig_corr)

with tab4:
    st.header("🤖 Interactive AI for Queries")
    st.info("Ask me a question about this dashboard, the models, or the dataset.")

    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if not gemini_api_key:
        st.warning("To use the AI, please set your Gemini API key in Streamlit secrets or as an environment variable.")
        st.stop()

    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('models/gemini-2.5-flash-lite')

    user_query = st.text_input("Your question:", "What is breast cancer?")

    if st.button("Ask AI"):
        if not user_query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    system_prompt = """
                    You are an AI assistant designed to answer questions about breast cancer and related topics like machine learning models, datasets, and medical terminology. 
                    Your primary role is to provide informative and educational content.
                    
                    **Crucially, you must not provide any medical advice, diagnosis, or treatment recommendations.**
                    Always end your responses with a strong disclaimer to consult a healthcare professional.
                    Be concise and accurate in your responses.
                    """
                    
                    full_query = f"{system_prompt}\n\nUser Question: {user_query}"
                    
                    response = model.generate_content(full_query)
                    ai_response = response.text
                    
                    st.markdown(f"**AI Assistant:** {ai_response}")

                    st.warning("""
                    **Disclaimer:** The information provided by this AI is for informational and educational purposes only and should not be considered medical advice. 
                    Always consult with a qualified healthcare professional for diagnosis, treatment, and medical decisions.
                    """)

                except Exception as e:
                    st.error(f"An error occurred: {e}")

# --- New Tab for Manual User Input based on report features ---
with tab5:
    st.header("📝 Report-Based Prediction")
    st.info("""
    Input key characteristics typically found in a breast cancer pathology report to get a prediction.
    """)
    
    st.subheader("Select a Model for Prediction")
    prediction_model_choice = st.selectbox("Choose a model:", list(results.keys()), key="report_predict_model")

    st.subheader("Input Tumor Characteristics")
    st.markdown("Please enter the values for the following features as they appear in the report. If a value is missing, use the default value provided.")

    # Create a form to group all the input widgets and a single submit button
    with st.form(key='report_form'):
        cols = st.columns(3)
        input_data = {}
        
        feature_descriptions = {
            'radius_mean': "Mean radius of the tumor",
            'texture_mean': "Mean texture (standard deviation of gray-scale values)",
            'perimeter_mean': "Mean perimeter of the tumor",
            'area_mean': "Mean area of the tumor",
            'smoothness_mean': "Mean smoothness (local variation in radius lengths)",
            'compactness_mean': "Mean compactness (perimeter^2 / area - 1)",
            'concavity_mean': "Mean concavity (severity of concave portions of the contour)",
            'concave points_mean': "Mean number of concave portions of the contour",
            'symmetry_mean': "Mean symmetry",
            'fractal_dimension_mean': "Mean fractal dimension ('coastline approximation' - 1)",
            'radius_se': "Standard error of the radius",
            'texture_se': "Standard error of the texture",
            'perimeter_se': "Standard error of the perimeter",
            'area_se': "Standard error of the area",
            'smoothness_se': "Standard error of the smoothness",
            'compactness_se': "Standard error of the compactness",
            'concavity_se': "Standard error of the concavity",
            'concave points_se': "Standard error of the concave points",
            'symmetry_se': "Standard error of the symmetry",
            'fractal_dimension_se': "Standard error of the fractal dimension",
            'radius_worst': "Worst or largest radius",
            'texture_worst': "Worst or largest texture",
            'perimeter_worst': "Worst or largest perimeter",
            'area_worst': "Worst or largest area",
            'smoothness_worst': "Worst smoothness",
            'compactness_worst': "Worst compactness",
            'concavity_worst': "Worst concavity",
            'concave points_worst': "Worst concave points",
            'symmetry_worst': "Worst symmetry",
            'fractal_dimension_worst': "Worst fractal dimension",
        }

        for i, feature in enumerate(feature_names):
            min_val = X[feature].min()
            max_val = X[feature].max()
            default_val = X[feature].mean()
            
            with cols[i % 3]:
                st.markdown(f"**{feature.replace('_', ' ').title()}**")
                st.markdown(f"<small>{feature_descriptions.get(feature, 'Feature description not available.')}</small>", unsafe_allow_html=True)
                input_data[feature] = st.number_input(
                    "",
                    min_value=float(min_val), 
                    max_value=float(max_val), 
                    value=float(default_val),
                    step= (max_val - min_val) / 100,
                    format="%.4f",
                    key=f'report_input_{feature}'
                )

        submit_button = st.form_submit_button(label="Get Diagnosis from Report Data")

    if submit_button:
        selected_model = results[prediction_model_choice]["model"]
        input_df = pd.DataFrame([input_data])

        scaled_input = scaler.transform(input_df)
        
        prediction = selected_model.predict(scaled_input)
        prediction_proba = selected_model.predict_proba(scaled_input)
        
        st.subheader("Prediction Result")

        if prediction[0] == 1:
            st.error("Based on the report data, the model predicts a **Malignant** tumor.")
        else:
            st.success("Based on the report data, the model predicts a **Benign** tumor.")

        st.subheader("Prediction Probabilities")
        st.write(f"Probability of Benign: **{prediction_proba[0][0]:.2f}**")
        st.write(f"Probability of Malignant: **{prediction_proba[0][1]:.2f}**")

        st.warning("""
        **Disclaimer:** This is a machine learning prediction and not a medical diagnosis. The results should be used for informational purposes only. Please consult a qualified healthcare professional for an accurate diagnosis and treatment plan.
        """)