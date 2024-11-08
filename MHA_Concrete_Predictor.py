import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import pandas as pd

# Streamlit input form for user to input values
st.title('Predicting the Mechanical Properties of Concrete Incoporating MHA and RWG')

# Dropdown for selecting the model type (Empirical or Machine Learning)
model_type = st.selectbox(
    'Select the model type',
    ['Empirical Model', 'Machine Learning Model']
)

# Input fields
cement_content = st.number_input('Cement Content (kg/m³)', min_value=453.00, max_value=565.00, step=1.00)
melon_husk_ash = st.number_input('Melon Husk Ash Content (%)', min_value=0.00, max_value=15.00, step=0.01)
recycled_waste_glass = st.number_input('Recycled Waste Glass Content (%)', min_value=0.00, max_value=5.00, step=0.01)
concrete_age = st.number_input('Concrete Age (Days)', min_value=7, max_value=56, step=1)

# Dropdown to select the concrete property (Compressive, Flexural, Tensile)
property_type = st.selectbox(
    'Select the concrete property to predict',
    ['Compressive Strength', 'Flexural Strength', 'Tensile Strength']
)

# Make prediction based on the selected model type
if model_type == 'Empirical Model':
    if st.button('Predict (Empirical Model)'):
        if property_type == 'Compressive Strength':
            # Calculate Compressive Strength using empirical formula
            compressive_strength = 17.3524 + (-0.0013) * cement_content + (-0.0252) * melon_husk_ash + (0.0265) * recycled_waste_glass + (0.3929) * concrete_age
            st.markdown(f"<h2 style='text-align: left; color: green; font-size: 32px;'>Compressive Strength: {compressive_strength:.3f} MPa</h2>", unsafe_allow_html=True)
        
        elif property_type == 'Flexural Strength':
            # Calculate Flexural Strength using empirical formula
            flexural_strength = 2.516 + (-0.00093) * cement_content + (-0.00425) * melon_husk_ash + (0.00518) * recycled_waste_glass + (0.045) * concrete_age
            st.markdown(f"<h2 style='text-align: left; color: green; font-size: 32px;'>Flexural Strength: {flexural_strength:.3f} MPa</h2>", unsafe_allow_html=True)
        
        elif property_type == 'Tensile Strength':
            # Calculate Tensile Strength using empirical formula
            tensile_strength = 1.485 + (0.00028) * cement_content + (0.00019) * melon_husk_ash + (-0.00047) * recycled_waste_glass + (0.043) * concrete_age
            st.markdown(f"<h2 style='text-align: left; color: green; font-size: 32px;'>Tensile Strength: {tensile_strength:.3f} MPa</h2>", unsafe_allow_html=True)

elif model_type == 'Machine Learning Model':
    # Load the model and the shared scaler
    scaler_file = 'scaler.pkl'  # Single shared scaler
    with open(scaler_file, 'rb') as scaler_file_obj:
        scaler = pickle.load(scaler_file_obj)

    # Based on the selected prediction type, load the corresponding model
    if property_type == 'Compressive Strength':
        model_file = 'Compressive_model.pkl'
    elif property_type == 'Flexural Strength':
        model_file = 'Flexure_model.pkl'
    else:  # Tensile Strength
        model_file = 'Tensile_model.pkl'

    # Load the selected model
    with open(model_file, 'rb') as model_file_obj:
        model = pickle.load(model_file_obj)

    # Make prediction on user input
    if st.button('Predict (Machine Learning Model)'):
        # Prepare the input data for prediction
        input_data = np.array([[cement_content, melon_husk_ash, recycled_waste_glass, concrete_age]])
        
        # Scale the inputs using the shared scaler
        input_scaled = scaler.transform(input_data)
        
        # Make the prediction using the loaded model
        pred_scaled = model.predict(input_scaled)
        
        # Denormalize the prediction (if the model was trained with normalization)
        prediction = pred_scaled[0]  # If you have scaling logic, inverse transform it
        
        # Display the prediction with larger font size
        st.markdown(f"<h2 style='text-align: left; color: green; font-size: 32px;'>Predicted {property_type}: {prediction:.3f} MPa</h2>", unsafe_allow_html=True)
        
        # SHAP values for feature importance
        st.subheader(f"{property_type} - SHAP Feature Importance")

        # Create a SHAP explainer (assuming the model is a tree-based model like XGBRegressor)
        explainer = shap.TreeExplainer(model)
        
        # SHAP values for the input
        shap_values = explainer.shap_values(input_scaled)

        # Plot the SHAP summary plot for feature importance
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_values, input_scaled, feature_names=['Cement Content', 'Melon Husk Ash', 'Recycled Waste Glass', 'Concrete Age'], plot_type="bar", show=False)
        
        # Display the SHAP plot in Streamlit
        st.pyplot(fig)
        
        # Add footnote
        st.markdown("""
            **Notes**: 
            1. This application predicts the concrete property based on input features using empirical and machine learning models.
            2. MHA: Melon Husk Ash, RWG: Recycled Waste Glass
            3. The fine aggregate, coarse aggregate and water contents of 628, 1506 and 216 kilogram per cubic meter, respectively were considered. 
        """)
        
        st.markdown("""
            **References**: 
            1. T. Chen, C. Guestrin,  XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, NY, USA, 2016: pp. 785–791. https://doi.org/10.1145/2939672.2939785.
            2. T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework, In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, New York, NY, USA, 2019: pp. 2623–2631. https://doi.org/10.1145/3292500.3330701.
        """)

# Footer with contact information
footer = """
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    text-align: center;
    padding: 10px;
    font-size: 12px;
    color: #6c757d;
}
</style>
<div class="footer">
    <p>© 2024 My Streamlit App. All rights reserved. | Temitope E. Dada | For Queries: <a href="mailto: T.Dada19@student.xjtlu.edu.cn"> T.Dada19@student.xjtlu.edu.cn</a></p>
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
