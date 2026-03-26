"""
Script to create Table S1: Model GWL crossing years and storyline assignments.
This table lists the 31 models for SSP5-8.5 and 25 models for SSP2-4.5,
detailing the year each model crosses +2C and +3C, and their storyline
assignment ("Increasing Frequency" or "Decreasing Frequency").
"""
import pandas as pd
import logging
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from storyline import StorylineAnalyzer
from config import Config
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def get_storyline(model_key, ext_list, non_ext_list):
    """Checks the storyline assignment of a model."""
    if ext_list and model_key in ext_list: 
        return "Increasing Frequency"
    if non_ext_list and model_key in non_ext_list: 
        return "Decreasing Frequency"
    return "Neutral / Other"

def save_dataframe_as_pdf(df, filename):
    """Saves a pandas DataFrame as a styled PDF table across multiple pages."""
    print(f"Rendering table to PDF: {filename}...")
    
    # Configuration for pagination
    rows_per_page = 32
    n_pages = (len(df) + rows_per_page - 1) // rows_per_page

    with PdfPages(filename) as pdf:
        for i in range(n_pages):
            start_idx = i * rows_per_page
            end_idx = min((i + 1) * rows_per_page, len(df))
            df_chunk = df.iloc[start_idx:end_idx]

            # Create figure with A4 proportions (approx 8.27 x 11.69 inches)
            fig, ax = plt.subplots(figsize=(12, 16))
            ax.axis('off')

            # Create the table
            table = ax.table(
                cellText=df_chunk.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center',
                colColours=['#f2f2f2'] * len(df.columns)
            )

            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.8) # Adjust cell height and width

            # Bold headers and add zebra striping
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#d9d9d9')
                elif row % 2 == 0:
                    cell.set_facecolor('#f9f9f9')

            # Add a title if it's the first page
            if i == 0:
                plt.title("Table S1: Model GWL Crossing Years and Storyline Assignments", 
                          fontsize=14, weight='bold', pad=20)
            
            # Add page number
            plt.figtext(0.95, 0.02, f"Page {i+1} of {n_pages}", ha='right', fontsize=9)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def main():
    # Configure minimal logging to stdout
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout)
    
    # Initialize the analyzer
    config = Config()
    analyzer = StorylineAnalyzer(config)
    
    records = []
    csv_filename = "S1_Model_Storylines_and_GWL_Years.csv"
    pdf_filename = "S1_Model_Storylines_and_GWL_Years.pdf"
    
    print(f"--- Generating Model Data for {pdf_filename} ---")
    
    # Process both scenarios
    for scenario in ['ssp585', 'ssp245']:
        print(f"Processing Scenario: {scenario.upper()}...")
        
        # This will load and process data for all models in the scenario
        # It takes time as it handles netCDF processing and GWL calculations
        cmip6_results = analyzer.analyze_cmip6_changes_at_gwl(scenario_to_process=scenario)
        
        if not cmip6_results:
            print(f"Warning: No results found for {scenario}")
            continue
            
        gwl_years = cmip6_results.get('gwl_threshold_years', {})
        
        # Get Storyline Classifications (Increasing vs Decreasing Frequency)
        # These are based on the return period changes of extreme low-flow events (30Q10)
        ext_models_w_2, non_ext_models_w_2, _ = analyzer.get_composite_extreme_models(cmip6_results, 2.0, '30Q10_low', 'Winter')
        ext_models_s_2, non_ext_models_s_2, _ = analyzer.get_composite_extreme_models(cmip6_results, 2.0, '30Q10_low', 'Summer')
        ext_models_w_3, non_ext_models_w_3, _ = analyzer.get_composite_extreme_models(cmip6_results, 3.0, '30Q10_low', 'Winter')
        ext_models_s_3, non_ext_models_s_3, _ = analyzer.get_composite_extreme_models(cmip6_results, 3.0, '30Q10_low', 'Summer')

        # Filter for models specifically in this scenario run
        models_in_scenario = sorted([m for m in gwl_years.keys() if m.endswith(scenario)])
        
        for model_key in models_in_scenario:
            model_name = model_key.split('_')[0]
            
            # Crossing Years
            y2 = gwl_years[model_key].get(2.0, "N/A")
            y3 = gwl_years[model_key].get(3.0, "N/A")
            
            # Storyline Assignments
            w2_st = get_storyline(model_key, ext_models_w_2, non_ext_models_w_2) if y2 != "N/A" else "N/A"
            s2_st = get_storyline(model_key, ext_models_s_2, non_ext_models_s_2) if y2 != "N/A" else "N/A"
            w3_st = get_storyline(model_key, ext_models_w_3, non_ext_models_w_3) if y3 != "N/A" else "N/A"
            s3_st = get_storyline(model_key, ext_models_s_3, non_ext_models_s_3) if y3 != "N/A" else "N/A"
            
            scen_name = "SSP5-8.5" if scenario == "ssp585" else "SSP2-4.5"
            records.append({
                "Model": model_name,
                "Scenario": scen_name,
                "GWL +2C Year": y2,
                "GWL +3C Year": y3,
                "Winter SL (+2C)": w2_st,
                "Summer SL (+2C)": s2_st,
                "Winter SL (+3C)": w3_st,
                "Summer SL (+3C)": s3_st,
            })
            
    # Create DataFrame
    df = pd.DataFrame(records)
    # Sort logically: Scenario then Model name
    df = df.sort_values(by=["Scenario", "Model"], ascending=[False, True])
    
    # Save CSV as backup
    df.to_csv(csv_filename, index=False)
    
    # Save PDF as requested
    save_dataframe_as_pdf(df, pdf_filename)
    
    print("-" * 40)
    print(f"Success! PDF Table created: {pdf_filename}")
    print(f"Total Models: {len(df)}")
    print("-" * 40)

if __name__ == "__main__":
    main()
