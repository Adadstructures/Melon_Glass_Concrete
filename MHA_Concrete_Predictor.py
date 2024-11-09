import streamlit as st
import numpy as np
import pickle
import optuna
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

# Streamlit input form for selecting action
st.title('Concrete Mix Predictor and Optimizer')

# Choose either to predict mechanical properties or optimize mix
option = st.radio(
    "Choose an option",
    ('Predict Mechanical Properties', 'Optimize Mix')
)

if option == 'Predict Mechanical Properties':
    # Predict Mechanical Properties Section
    st.header('Predicting the Mechanical Properties of Concrete Incorporating MHA and RWG')

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
            prediction = pred_scaled[0]

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

            
# Optimization Section
elif option == 'Optimize Mix':
    st.header('Optimizing Concrete Mix for Target Strengths and Carbon Footprint')

    # Dropdown to select the concrete property to optimize (Compressive, Flexural, Tensile)
    property_type_for_optimization = st.selectbox(
        'Select the concrete property to optimize',
        ['Compressive Strength', 'Flexural Strength', 'Tensile Strength']
    )

    # Input fields for target values
    st.subheader('Specify Target Strengths and Carbon Footprint')
    
    # Set the appropriate range based on the selected property type
    if property_type_for_optimization == 'Compressive Strength':
        min_value, max_value, step = 13.72, 37.43, 0.1
    elif property_type_for_optimization == 'Flexural Strength':
        min_value, max_value, step = 2.01, 4.57, 0.1
    else:  # Tensile Strength
        min_value, max_value, step = 1.45, 4.21, 0.1
    
    # Input field for target strength with dynamic range
    target_strength = st.number_input(f'Target {property_type_for_optimization} (MPa)', min_value=min_value, max_value=max_value, step=step)

    target_carbon_footprint = st.number_input('Target Carbon Footprint (kg CO₂/m³)', min_value=100.0, max_value=550.0, step=1.0)

    # Input fields for minimum percentage target values for MHA and RWG
    min_mha_percentage = st.number_input('Minimum Target MHA Percentage (%)', min_value=0.0, max_value=15.0, step=0.1)
    min_rwg_percentage = st.number_input('Minimum Target RWG Percentage (%)', min_value=0.0, max_value=5.0, step=0.1)

    @st.cache_resource
    def load_models():
        with open('Compressive_model.pkl', 'rb') as file:
            compressive_model = pickle.load(file)
        with open('Flexure_model.pkl', 'rb') as file:
            flexural_model = pickle.load(file)
        with open('Tensile_model.pkl', 'rb') as file:
            tensile_model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return compressive_model, flexural_model, tensile_model, scaler
    
    compressive_model, flexural_model, tensile_model, scaler = load_models()

    # Constants for aggregates and water
    water = 216  # kg
    fineaggregate = 628  # kg
    coarseaggregate = 1506  # kg
    
    # Carbon footprint values
    carbon_footprints = {
        'cement': 0.82,
        'mha': 0.07574,
        'rwg': 0.02794,
        'water': 0.000155,
        'coarseaggregate': 0.0408,
        'fineaggregate':0.0139
    }

    # Densities for materials
    densities = {
        'cement': 1440,  # kg/m³
        'mha': 1500,     # kg/m³
        'rwg': 2650,     # kg/m³
        'coarseaggregate': 1800,  # kg/m³
        'fineaggregate': 1600,    # kg/m³
        'water': 1000     # kg/m³
    }

    # Function to calculate carbon footprint
    def calculate_carbon_footprint(cement, mha, rwg, water, coarseaggregate, fineaggregate):
        return (
            cement * carbon_footprints['cement'] + 
            mha * carbon_footprints['mha'] + 
            rwg * carbon_footprints['rwg'] + 
            water * carbon_footprints['water'] +
            coarseaggregate * carbon_footprints['coarseaggregate'] +
            fineaggregate * carbon_footprints['fineaggregate']
        )

    # Function to calculate the volume of the mix
    def calculate_volume(cement, mha, rwg, water, coarseaggregate, fineaggregate):
        total_binder = cement + mha + rwg
        total_aggregate = coarseaggregate + fineaggregate
        return total_binder + total_aggregate + water

    # Function to predict strength based on selected property type
    def predict_strength(model, scaler, cement, mha, rwg, age):
        input_data = np.array([[cement, mha, rwg, age]])
        input_scaled = scaler.transform(input_data)
        return model.predict(input_scaled)[0]

    # Optimization objective function
    def objective(trial, target_strength, target_carbon_footprint, min_mha_target, min_rwg_target, property_type_for_optimization):
        # Sample hyperparameters from the trial object
        cement = trial.suggest_float('cement', 200.0, 600.0)
        
        # Calculate the minimum MHA and RWG values based on percentage
        min_mha_value = (min_mha_target / 100.0) * cement
        min_mha_value = min(min_mha_value, 84.0)
        min_rwg_value = (min_rwg_target / 100.0) * cement
        min_rwg_value = min(min_rwg_value, 28.0)

        mha = trial.suggest_float('mha', min_mha_value, 84.0)  # Ensure MHA is at least the minimum target
        rwg = trial.suggest_float('rwg', min_rwg_value, 28.0)  # Ensure RWG is at least the minimum target
        
        # Calculate total volume
        total_volume = calculate_volume(cement, mha, rwg, water, coarseaggregate, fineaggregate)
        
        # Predict strength using the selected model
        if property_type_for_optimization == 'Compressive Strength':
            strength_pred = predict_strength(compressive_model, scaler, cement, mha, rwg, age)
        elif property_type_for_optimization == 'Flexural Strength':
            strength_pred = predict_strength(flexural_model, scaler, cement, mha, rwg, age)
        else:
            strength_pred = predict_strength(tensile_model, scaler, cement, mha, rwg, age)
        
        # Calculate carbon footprint
        total_carbon_footprint = calculate_carbon_footprint(cement, mha, rwg, water, coarseaggregate, fineaggregate)
        
        # Strength penalty: heavily penalize deviations from the target
        strength_diff = (target_strength - strength_pred) ** 5 
        
        # Carbon footprint penalty: heavily penalize deviations from the target
        carbon_diff = (total_carbon_footprint - target_carbon_footprint) ** 50
        
        # Cement penalty: incentivize reducing cement usage
        cement_penalty = (cement / 565.0) ** 50
        
        # Maximize the use of MHA and RWG, by adding a reward for higher values
        mha_reward = (mha / 85.0)  ** 5
        rwg_reward = (rwg / 30.0)  ** 5
        
        # Objective: Minimize penalties for strength, carbon footprint, and cement use
        # while rewarding the use of MHA and RWG
        objective_value = (strength_diff + carbon_diff - cement_penalty + (mha_reward + rwg_reward))
        
        return objective_value

    # Streamlit Inputs for age
    age_options = [7, 14, 21, 28, 56]
    age = st.selectbox("Select Concrete Age (days)", age_options)

    # When the optimize button is pressed
    if st.button('Optimize'):
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, target_strength, target_carbon_footprint, min_mha_percentage, min_rwg_percentage, property_type_for_optimization), n_trials=100)
        
        # Get best parameters
        best_params = study.best_params
        cement = best_params['cement']
        mha = best_params['mha']
        rwg = best_params['rwg']
        
        # Calculate the total carbon footprint for the optimized mix
        total_carbon_footprint = calculate_carbon_footprint(cement, mha, rwg, water, coarseaggregate, fineaggregate)
        
        # Calculate Melon Husk Ash and Recycled Waste Glass as a percentage of Cement
        mha_percentage = (mha / cement) * 100
        rwg_percentage = (rwg / cement) * 100
        
        # Predict the final strength
        if property_type_for_optimization == 'Compressive Strength':
            strength_pred = predict_strength(compressive_model, scaler, cement, mha, rwg, age)
        elif property_type_for_optimization == 'Flexural Strength':
            strength_pred = predict_strength(flexural_model, scaler, cement, mha, rwg, age)
        else:
            strength_pred = predict_strength(tensile_model, scaler, cement, mha, rwg, age)
        
        st.write(f"Optimized Cement: {cement:.2f} kg/m³")
        st.write(f"Optimized MHA: {mha:.2f} kg/m³ ({mha_percentage:.2f}% of Cement)")
        st.write(f"Optimized RWG: {rwg:.2f} kg/m³ ({rwg_percentage:.2f}% of Cement)")
        st.write(f"Target Strength: {target_strength} MPa")
        st.write(f"Predicted Strength: {strength_pred:.2f} MPa")
        st.write(f"Carbon Footprint: {total_carbon_footprint:.2f} kg CO₂/m³")

        
        # # Show prediction accuracy
        # prediction_accuracy = 1 - abs(optimized_strength - target_strength) / target_strength
        # st.write(f"Prediction Accuracy: {prediction_accuracy:.2%}")

    
# Add footnote
st.markdown("""
            **Notes**: 
            1. This application predicts the concrete property based on input features using empirical and machine learning models.
            2. MHA: Melon Husk Ash, RWG: Recycled Waste Glass
            3. The fine aggregate, coarse aggregate and water contents of 628, 1506, and 216 kg/m³, respectively, were considered.
            """)

st.markdown("""
        **References**: 
        1. T. Chen, C. Guestrin, XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, New York, NY, USA, 2016: pp. 785–791. https://doi.org/10.1145/2939672.2939785.
        2. T. Akiba, S. Sano, T. Yanase, T. Ohta, M. Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, Association for Computing Machinery, New York, NY, USA, 2019: pp. 2623–2631. https://doi.org/10.1145/3292500.3330701.
        """)