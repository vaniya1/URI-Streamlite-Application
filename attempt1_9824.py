#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st
import sys
from sys import stdout
import lasio
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas import set_option

set_option("display.max_rows", 10)
pd.options.mode.chained_assignment = None
lithology_dict = {
    3120: "SSH", 3000: "S", 7100: "SH", 1600: "SHS", 3360: "SSH", 6860: "SH",
    2180: "S", 1860: "SHS", 2060: "SH", 2480: "SSH", 7200: "LSSH", 6780: "LS",
    2140: "SSH", 6580: "LSSH", 3100: "SSH", 7280: "LS", 1400: "SSS", 2360: "SSH",
    1340: "S", 2800: "S", 7440: "LS", 6960: "LS", 3340: "SSH", 7320: "LS",
    7170: "SH", 3500: "S", 2440: "SSH", 2400: "SSH", 1000: "LS", 2420: "S",
    2600: "SSS", 2000: "S", 7200: "LS", 2810: "S"
}
# Define mnemonics and units
DT_mnemonic = ['DT']
DT_units = ['US/FT']
DEPT_mnemonic = ['DEPT', 'DEPTH']
DEPT_units = ['FT', 'M']
DTS_mnemonic = ['DTS', 'DTSH', "DELTA", "TT", "AC", "SLOW"]
DTS_units = ['US/FT']
RHOB_mnemonic = ['RHOB', "RHOZ", "DEN", "DENB", "DENC", "DENCDL", "HRHO", "HRHOB", "ZDEN", "ZDENS", "ZDNCS", "ZDNC", "HDEN", "DENF", "DENN"]
RHOB_units = ['G/CC', 'kg/m3', 'g/cm3']
NPHI_mnemonic = ['NPHI', 'NPRS', 'NPORS', "NPHIS", "HNPOS", "HNPO_SS", "NPOR_S", "NPOR_SS", "HNTP_SS", "TNPH_SAN", "HNTP_SAN", "HNPO_SAN", "NPOR_SAN", "NPorSand"]
NPHI_units = ['V/V']


def get_formation_and_tops(las, formationtops):
    well_name = las.well['API'].value
    filtered_data = formationtops[formationtops['API_UWI_12'] == well_name]
    if filtered_data.empty:
        return las, pd.DataFrame(columns=['FormationName', 'MD_Top', 'FormationCode', 'MD_Bottom', 'LithoCode'])
    formation_details = filtered_data[['FormationName', 'MD_Top', 'FormationCode']]
    formation_details = formation_details.sort_values(by='MD_Top', ascending=True)
    formation_details['MD_Bottom'] = formation_details['MD_Top'].shift(-1)
    formation_details['LithoCode'] = formation_details['FormationCode'].map(lithology_dict).fillna('None')
    depth_unit = las.curves['DEPT'].unit.lower() 
    if depth_unit == 'ft':
        max_depth = las.depth_ft.max()
    elif depth_unit == 'm':
        max_depth = las.depth_m.max()
    else:
        raise ValueError(f"Unexpected depth unit: {depth_unit}")
    formation_details.at[formation_details.index[-1], 'MD_Bottom'] = max_depth
    return las, formation_details

def select_curve(las, mnemonic_keywords, units):
    matched_curves = []
    for curve in las.curves:
        for keyword in mnemonic_keywords:
            for unit in units:
                if keyword in curve.mnemonic and curve.unit == unit:
                    matched_curves.append(curve)
    if len(matched_curves) > 1:
        # Streamlit doesn't handle console input. Consider using st.selectbox for multiple options.
        choices = [f"{i+1}: Mnemonic: {curve.mnemonic}, Unit: {curve.unit}" for i, curve in enumerate(matched_curves)]
        choice = st.selectbox("Select a curve", options=choices + ["0: Skip"], index=0)
        if choice.startswith("0"):
            return None
        index = int(choice.split(":")[0]) - 1
        return matched_curves[index].mnemonic
    elif len(matched_curves) == 1:
        return matched_curves[0].mnemonic
    matched_unit_curves = []
    for curve in las.curves:
        if curve.unit in units:
            matched_unit_curves.append(curve)
    if matched_unit_curves:
        choices = [f"{i+1}: Mnemonic: {curve.mnemonic}, Unit: {curve.unit}" for i, curve in enumerate(matched_unit_curves)]
        choice = st.selectbox("Select a curve", options=choices + ["0: Skip"], index=0)
        if choice.startswith("0"):
            return None
        index = int(choice.split(":")[0]) - 1
        return matched_unit_curves[index].mnemonic
    return None

def Calculate_mechanical_Prop(las, formation_details):
    DEPT = select_curve(las, DEPT_mnemonic, DEPT_units)
    DT = select_curve(las, DT_mnemonic, DT_units)
    RHOB = select_curve(las, RHOB_mnemonic, RHOB_units)
    DTS = select_curve(las, DTS_mnemonic, DTS_units)
    NPHI = select_curve(las, NPHI_mnemonic, NPHI_units)
    selected_logs = {}
    if DEPT: selected_logs['DEPT'] = DEPT
    if DT: selected_logs['DT'] = DT
    if RHOB: selected_logs['RHOB'] = RHOB
    if DTS: selected_logs['DTS'] = DTS
    if NPHI: selected_logs['NPHI'] = NPHI    
    if not selected_logs:
        raise ValueError("No valid curves selected. Cannot proceed with mechanical property calculations.")
    WellLogDf = las.df().reset_index()  
    try:
        logData = WellLogDf[selected_logs.values()].copy()
    except KeyError as e:
        raise KeyError(f"Error selecting curves from LAS file: {e}")  
    logData.columns = selected_logs.keys()
    # unit conversions:
    if DT_units == 'US/FT':
        logData['DT'] *= 3.281 
    if DTS_units == 'US/FT' and 'DTS' in logData.columns:
        logData['DTS'] *= 3.281
    if RHOB_units == 'KG/M3':
        logData['RHOB'] *= 0.001  
    # if have DT, use VP and do the following calculations
    if 'DT' in logData.columns:
        logData['VP'] = 304.8 / logData['DT']  
    if 'DTS' in logData.columns and logData['DTS'].notna().any():
        logData['VS'] = 304.8 / logData['DTS']
    elif 'VP' in logData.columns:
        logData['VS'] = None
        # No VS
        for index, row in logData.iterrows():
            matching_rows = formation_details.loc[
                (formation_details['MD_Top'] <= row['DEPT']) & (row['DEPT'] <= formation_details['MD_Bottom']),
                'LithoCode']
            if not matching_rows.empty:
                Lith = matching_rows.values[0]
            else:
                Lith = 'None' 
            if Lith in ['SS', 'SOG', 'S', 'SSS', 'SO', 'SCO', 'SG']:
                logData.at[index, 'VS'] = (0.8042 * row['VP'] - 855.9)/1000
            elif Lith == 'LS':
                logData.at[index, 'VS'] = (1.0168 * row['VP'] - 0.00005509 * row['VP']**2 - 1030.5)/1000
            elif Lith == 'DOL':
                logData.at[index, 'VS'] = (0.5832 * row['VP'] - 77.76)/1000
            elif Lith in ['SHOG', 'SH','SSH','SHS']:
                logData.at[index, 'VS'] = (0.77 * row['VP'] - 867.4)/1000    
    if 'VP' in logData.columns and 'VS' in logData.columns:
        logData['VPVS'] = logData['VP'] / logData['VS']  
    logData['Formation'] = None 
    logData['Lithology'] = None
    logData['Formation Code'] = None
    for index, row in logData.iterrows():
        depth = row['DEPT']
        mask = (formation_details['MD_Top'] <= depth) & (depth <= formation_details['MD_Bottom'])
        matching_row = formation_details[mask]
        if not matching_row.empty:
            logData.at[index, 'Formation Code'] = matching_row.iloc[0]['FormationCode']
            logData.at[index, 'Lithology'] = matching_row.iloc[0]['LithoCode']
            logData.at[index, 'Formation'] = matching_row.iloc[0]['FormationName']
    logData = logData[logData['Lithology'].notna()]  
    MechProp = pd.DataFrame()
    MechProp['DEPT'] = logData['DEPT'].copy()
    MechProp['Formation Code'] = logData['Formation Code'].copy()
    MechProp['Formation'] = logData['Formation'].copy()
    MechProp['G'] = (logData['RHOB']) * (logData['VP'] ** 2)
    MechProp['YM'] = ((logData['RHOB']) * logData['VS']**2 * (3*logData['VP']**2) - 4*logData['VS']**2) / (logData['VP']**2 - logData['VS']**2)
    MechProp['PRa'] = ((logData['VP']**2) - 2*(logData['VS']**2)) / (2*((logData['VP']**2) - (logData['VS']**2)))
    MechProp['K'] = (logData['RHOB'] * 0.001) * ((logData['VP']**2) - ((4/3) * (logData['VS']**2)))
    UCS, Coh, FA = [], [], []
    for index, row in logData.iterrows():
        Lith = row['Lithology']
        if Lith == 'SHOG' or Lith == 'SH':
            ucs = 1.35 * (304.8/row['DT']) * ((0.57 * row['G']) + 0.83)
            coh = (0.023* (304.8/row['DT'])) * ((0.57 * row['G']) + 0.83) * 0.045
            fa = 0.013 * ucs
        elif Lith == 'SOG' or Lith == 'S' or Lith == 'SSS' or Lith == 'SS':
            ucs = 1.35 * (304.8/row['DT']) * ((0.64 * row['G']) + 0.94)
            coh = (0.023* (304.8/row['DT'])) * ((0.64 * row['G']) + 0.94) * 0.045
            fa = 0.013 * ucs
        elif Lith == 'LS':
            ucs = 1.35 * (304.8/row['DT']) * ((0.65 * row['G']) + 0.90)
            coh = (0.023* (304.8/row['DT'])) * ((0.65 * row['G']) + 0.90) * 0.045
            fa = 0.013 * ucs
        elif Lith == 'DOL':
            ucs = 1.35 * (304.8/row['DT']) * ((0.69 * row['G']) + 1.10)
            coh = (0.023* (304.8/row['DT'])) * ((0.69 * row['G']) + 1.10) * 0.045
            fa = 0.013 * ucs
        else:
            ucs, coh, fa = np.nan, np.nan, np.nan
        UCS.append(ucs)
        Coh.append(coh)
        FA.append(fa)
    MechProp['UCS'] = UCS
    MechProp['Cohesion'] = Coh
    MechProp['FrictionAngle'] = FA
    return MechProp, logData

def plot_lithology(logData, MechProp):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    ax1.plot(logData['DEPT'], logData['RHOB'], 'b-', label='Density (RHOB)')
    ax2.plot(logData['DEPT'], logData['DT'], 'r-', label='Sonic (DT)')
    ax1.set_xlabel('Depth (m)')
    ax1.set_ylabel('Density (RHOB)', color='b')
    ax2.set_ylabel('Sonic (DT)', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid()
    ax1.set_title('Lithology')
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(MechProp['DEPT'], MechProp['UCS'], c=MechProp['Formation Code'].astype('category').cat.codes, cmap='viridis')
    ax.set_xlabel('Depth (m)')
    ax.set_ylabel('UCS')
    ax.set_title('UCS by Formation')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Formation Code')
    st.pyplot(fig)

def display_mechanical_properties_table(MechPropMean):
    # Creating a DataFrame for the mean values by grouping by formation
    mean_properties_df = MechPropMean.groupby('Formation Code').mean().reset_index()
    # Creating a DataFrame for all mechanical properties
    table_df = pd.DataFrame({
        'Formation Code': mean_properties_df['Formation Code'],
        'G (Modulus of Rigidity)': mean_properties_df['G'],
        'YM (Young\'s Modulus)': mean_properties_df['YM'],
        'PRa (Poisson\'s Ratio)': mean_properties_df['PRa'],
        'K (Bulk Modulus)': mean_properties_df['K'],
        'UCS (Unconfined Compressive Strength)': mean_properties_df['UCS'],
        'Coh (Cohesion)': mean_properties_df['Coh'],
        'FA (Friction Angle)': mean_properties_df['FA']
    })
    st.write(table_df)
    return table_df

def main():
    st.title("Mechanical Properties of Selected Curves and LAS File")
    
    activities = ["Table of Mech. Properties", "Plots of Mechanical Properties"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    st.subheader(f"Activity Selected: {choice}")

    # File uploading for both LAS and Formation Table
    las_file = st.file_uploader("Upload a LAS File", type=["LAS", "las"])
    formation_file = st.file_uploader("Upload a Formation Table File", type=["csv"])

    if las_file and formation_file:
        las = lasio.read(las_file)
        formationtops = pd.read_csv(formation_file)
        las, formation_details = get_formation_and_tops(las, formationtops)
        MechProp, logData = Calculate_mechanical_Prop(las, formation_details)

        # Show the user the matched curves for their selection
        st.subheader("Matched Curves with Units")
        st.write("Unit found but no mnemonic keyword matched.")
        st.write("Curves with matched unit(s) (US/FT):")
        st.write("1: Mnemonic: DT_E, Unit: US/FT")

        # Let the user choose the curve
        curve_selection = st.text_input("Select a curve (1-1) or 0 to skip:", "1")

        if curve_selection == "1":
            # Option for Table of Mechanical Properties
            if choice == "Table of Mech. Properties":
                MechPropMean = MechProp.groupby('Formation Code').mean().reset_index()
                st.subheader("Mechanical Properties Table")
                display_mechanical_properties_table(MechPropMean)

            # Option for Plots of Mechanical Properties
            elif choice == "Plots of Mechanical Properties":
                st.subheader("Mechanical Properties Plots")
                plot_lithology(logData, MechProp)
        else:
            st.write("No valid curve selected.")

if __name__ == '__main__':
    main()
# In[11]:





# In[ ]:




