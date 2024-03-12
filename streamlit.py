import streamlit as st

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import truncnorm
import random
import json

columns = ['Urea', 'Cr', 'HbA1c', 'BGL', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']

def get_samples(lower_bound:float, upper_bound:float, mean:float, sd:float, size:int)->np.ndarray:
    """Performs random sampling on truncated standard distribution. Truncation is defined by `lower_bound` and `upper_bound`.

    Args:
        `lower_bound` (float): lower limit on the distribution
        `upper_bound` (float): upper limit on the distribution
        `mean` (float): mean of the standard distribution
        `sd` (float): standard deviation of the standard distribution
        `size` (int): number of points to sample

    Raises:
        `e`: generic errors/exceptions

    Returns:
        `np.ndarray`: array of generated data
    """    
    try:
        # (lower_bound, upper_bound) = (upper_bound, lower_bound) if lower_bound>upper_bound else (lower_bound, upper_bound)
        z_score_low = (lower_bound - mean) /sd
        z_score_up = (upper_bound - mean) /sd
        data = truncnorm.rvs(a=z_score_low, b=z_score_up, loc=mean, scale=sd, size=size)
        # data.sort()
        return data
    except Exception as e:
        raise e

def generate_data(reference: dict, parameter: str, stage: str, thresh: float, size: int=25) -> np.ndarray:
    """Generates data for the given parameter `parameter` using the confidence intervals in `reference` dictionary.

    Args:
        `reference` (dict): dictionary containing mean and standard deviation of each parameter
        `parameter` (str): the parameter for which data is being generated
        `stage` (str): current stage as per the input data
        `thresh` (float): upper limit for data generation
        `size` (int, optional): number of samples to generate

    Raises:
        `e`: generic errors/exceptions

    Returns:
        `np.ndarray`: generated data with `size` number of samples
    """  
    try:  
        if stage in ["Diabetes", "Pre-Diabetes", "No Diabetes"]:
            mean = reference['STAGE-ONE'][parameter]['mean']
            std = reference['STAGE-ONE'][parameter]['std']
            s1_lb = mean - 2*std if (mean - 2*std) > 0 else mean
            if stage != "No Diabetes":
                s1_ub = mean + 2*std if ((mean + 2*std) < thresh) else thresh
            else:
                s1_ub = thresh
            print(f"1:::{parameter} - mean: {mean}, std: {std}, upper limit: {s1_ub}, lower limit: {s1_lb}, thresh: {thresh}")
            stage1_values = get_samples(lower_bound=s1_lb, upper_bound=s1_ub, mean=mean, sd=std, size=size)
        #----------------------------------------------
        if stage in ["Diabetes", "Pre-Diabetes"]:
            mean = reference['STAGE-TWO'][parameter]['mean']
            std = reference['STAGE-TWO'][parameter]['std']
            s2_lb = s1_ub - 2*std if ((mean - 2*std) < (s1_ub - 2*std)) else mean - 2*std
            if stage != "Pre-Diabetes":
                s2_ub = mean + 2*std if ((mean + 2*std) < thresh) else thresh
            else:
                s2_ub = thresh
            print(f"2:::{parameter} - mean: {mean}, std: {std}, upper limit: {s2_ub}, lower limit: {s2_lb}, thresh: {thresh}")
            stage2_values = get_samples(lower_bound=s2_lb, upper_bound=s2_ub, mean=mean, sd=std, size=size)
        #------------------------------------------------
        if stage in ["Diabetes"]:
            mean = reference['STAGE-THREE'][parameter]['mean']
            std = reference['STAGE-THREE'][parameter]['std']
            s3_lb = s2_ub - 2*std if ((mean - 2*std) < (s2_ub - 2*std)) else mean - 2*std
            s3_ub = mean + 0.8*std if stage != "Diabetes" else thresh
            print(f"3:::{parameter} - mean: {mean}, std: {std}, upper limit: {s3_ub}, lower limit: {s3_lb}, thresh: {thresh}")
            stage3_values = get_samples(lower_bound=s3_lb, upper_bound=s3_ub, mean=mean, sd=std, size=size)

        if stage=="Diabetes":
            values = np.concatenate([stage1_values, stage2_values, stage3_values])
        elif stage=="Pre-Diabetes":
            values = np.concatenate([stage1_values, stage2_values])
        else:
            values = stage1_values
        values.sort()
        return values
    except Exception as e:
        raise e

def history_generator(current_data: dict, size: int) -> pd.DataFrame:
    """Generates synthetic data keeping `current_data` values as upper thresholds

    Args:
        `current_data` (dict): input data to be used as reference or upper threshold
        `size` (int): number of samples to generate

    Raises:
        `e`: generic errors/exceptions

    Returns:
        `pd.DataFrame`: generated data in dataframe format
    """    
    try:
        age = current_data.get("Age", random.randint(18, 65))
        stage = current_data.get("Health_Status", None)
        end_date = current_data.get("Date", f'2024-{random.randint(1, 12)}-{random.randint(1, 28)}')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        if stage == "Diabetes":
            adj_size = size//3 
            health_status = ["No Diabetes"]*adj_size + ["Pre-Diabetes"]*adj_size + ["Diabetes"]*adj_size 
        elif stage == "Pre-Diabetes":
            adj_size = size//2 
            health_status = ["No Diabetes"]*adj_size + ["Pre-Diabetes"]*adj_size
        else:
            adj_size = size 
            health_status = ["No Diabetes"]*adj_size 

            
        # determine the age group for fetching confidence intervals
        if age in range(18, 36):
            path = 'reference\\young.json'
        elif age in range(36, 46):
            path = 'reference\\middle.json'
        else:
            path = 'reference\\old.json'
        
        print(f'Age is {age}, chosen path {path}')

        # fetching confidence intervals of all parameters per stage
        with open(path) as f:
            dictt = json.load(f)

        data = pd.DataFrame(columns=columns)
        for col in columns:
            limit = current_data.get(col, None)
            data[col] = generate_data(reference=dictt, parameter=col, stage=stage, thresh=limit, size=adj_size)
            data[col] = data[col].apply(lambda x: round(x, 2))
        data['Health_Status'] = health_status
            
        dates = []
        dates.append(end_date.date())
        for i in range(data.shape[0]-1):
            prev_date = dates[-1]
            new_date = prev_date - timedelta(days=30)
            dates.append(new_date)
        dates.reverse()
        data['Date'] = dates

        birth_date = datetime.strptime(f'{end_date.year - age}-12-12', '%Y-%m-%d').date()
        data['Age'] = data['Date'].apply(lambda x: (x - birth_date).days//365 )
        data = pd.concat([data, pd.DataFrame(current_data, index=[data.shape[0]])])
        return data[['Date', 'Age', 'Health_Status', 'Urea', 'Cr', 'HbA1c', 'BGL', 'Chol', 'TG', 'HDL', 'LDL', 'VLDL', 'BMI']]
    except Exception as e:
        raise e

# Streamlit app
def main():
    # Title of the app
    st.title('Synthetic Health Data Generator')

    # Get user input for parameters
    st.subheader("Input Parameters")
    date = st.text_input("Date (YYYY-MM-DD)")
    age = st.text_input("Age")
    health_status = st.selectbox("Health Status", ["No Diabetes", "Pre-Diabetes", "Diabetes"])
    urea = st.text_input("Urea")
    cr = st.text_input("Cr")
    hba1c = st.text_input("HbA1c")
    chol = st.text_input("Chol")
    tg = st.text_input("TG")
    hdl = st.text_input("HDL")
    ldl = st.text_input("LDL")
    vldl = st.text_input("VLDL")
    bmi = st.text_input("BMI")
    bgl = st.text_input("BGL")
    date_type = type(date)
    st.write(date_type)
    size = st.number_input("Sample Size", min_value=1, max_value=1000, step=1, value=100)

    input_data = {
        "Date": date,
        "Age": age,
        "Health_Status": health_status,
        "Urea": urea,
        "Cr": cr,
        "HbA1c": hba1c,
        "BGL": bgl,
        "Chol": chol,
        "TG": tg,
        "HDL": hdl,
        "LDL": ldl,
        "VLDL": vldl,
        "BMI": bmi
    }

    # Function to generate synthetic data
    def generate_data(input_data, size):
        try:
            # Convert Date string to datetime object
            input_data['Date'] = datetime.strptime(input_data['Date'], '%Y-%m-%d')
            
            # Generate synthetic data
            data = history_generator(current_data=input_data, size=size)
            return data
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Button to generate data
    if st.button("Generate Data"):
        data = generate_data(input_data, size)
        st.dataframe(data)

        # Download link for the generated CSV file
        csv = data.to_csv(index=False)
        st.download_button("Download CSV", csv, "synthetic_data.csv", "text/csv")

if __name__ == "__main__":
    main()
