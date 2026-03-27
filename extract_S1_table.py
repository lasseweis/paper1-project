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

def _render_styled_table(ax, df_chunk, full_columns, title, is_first_page=True):
    """Helper to render a styled table on a given axes."""
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=df_chunk.values,
        colLabels=full_columns,
        cellLoc='center',
        loc='center',
        # colColours=['#f2f2f2'] * len(full_columns)
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(7.5) 
    table.scale(1.5, 1.8) # Increased width scale, reduced height scale

    # Professional styling for research papers
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.4)
        
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='black')
            cell.set_facecolor('#E0E0E0')
            cell.visible_edges = 'TB' 
        else:
            # Body styling
            cell.visible_edges = 'B' 
            if row % 2 == 0:
                cell.set_facecolor('#FDFDFD')
            else:
                cell.set_facecolor('#F5F5F5')
            
            if cell.get_text().get_text() == "N/A":
                cell.get_text().set_color('#888888')

    # Title is now handled by suptitle in the calling function
    return table

def save_dataframe_as_pdf(df, filename, title):
    """Saves a pandas DataFrame as a styled PDF table across multiple pages."""
    print(f"Rendering table to PDF: {filename}...")
    
    # Configuration for pagination
    rows_per_page = 28 # Back to 28 since height scale is reduced
    n_pages = (len(df) + rows_per_page - 1) // rows_per_page

    with PdfPages(filename) as pdf:
        for i in range(n_pages):
            start_idx = i * rows_per_page
            end_idx = min((i + 1) * rows_per_page, len(df))
            df_chunk = df.iloc[start_idx:end_idx]

            # True Landscape A4
            fig, ax = plt.subplots(figsize=(14, 10)) 
            _render_styled_table(ax, df_chunk, df.columns, title, is_first_page=(i==0))
            
            if i == 0:
                fig.suptitle(title, fontsize=16, fontweight='bold', y=0.94)
            
            # Adjusted margins to prevent overlap and center the table better
            plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.95)
            
            plt.figtext(0.95, 0.05, f"Page {i+1} of {n_pages}", ha='right', fontsize=8, color='gray')
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

def save_dataframe_as_png(df, filename, title):
    """Saves the entire DataFrame as a single, tall PNG image."""
    print(f"Rendering table to PNG: {filename}...")
    
    # Wider figure for PNG to prevent overlaps
    fig_height = len(df) * 0.4 + 3
    fig, ax = plt.subplots(figsize=(16, fig_height))
    
    _render_styled_table(ax, df, df.columns, title, is_first_page=True)
    
    # Use suptitle for PNG as well, adjusting y based on height
    # For very tall figures, y=0.98 is safer to keep it near the top
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    plt.subplots_adjust(top=0.94, bottom=0.02, left=0.05, right=0.95)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    # Configure minimal logging to stdout
    logging.basicConfig(level=logging.ERROR, stream=sys.stdout)
    
    # Initialize the analyzer
    config = Config()
    analyzer = StorylineAnalyzer(config)
    
    print(f"--- Generating Model Data for S1 Table ---")
    
    # Process both scenarios separately
    for scenario in ['ssp585', 'ssp245']:
        print(f"Processing Scenario: {scenario.upper()}...")
        records = []
        
        # This will load and process data for all models in the scenario
        cmip6_results = analyzer.analyze_cmip6_changes_at_gwl(scenario_to_process=scenario)
        
        if not cmip6_results:
            print(f"Warning: No results found for {scenario}")
            continue
            
        gwl_years = cmip6_results.get('gwl_threshold_years', {})
        
        # Get Storyline Classifications
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
            
            record = {
                "Model": model_name,
                "Scenario": scen_name,
                "GWL +2C Year": y2,
                "Winter SL (+2C)": w2_st,
                "Summer SL (+2C)": s2_st,
            }
            
            # Add +3C details only for SSP5-8.5
            if scenario == 'ssp585':
                record.update({
                    "GWL +3C Year": y3,
                    "Winter SL (+3C)": w3_st,
                    "Summer SL (+3C)": s3_st,
                })
            
            records.append(record)
            
        # Create DataFrame for this scenario
        df = pd.DataFrame(records)
        df = df.sort_values(by=["Model"], ascending=True)
        
        # File naming
        base_filename = f"S1_{scenario}_Model_Storylines_and_GWL_Years"
        csv_filename = f"{base_filename}.csv"
        pdf_filename = f"{base_filename}.pdf"
        png_filename = f"{base_filename}.png"
        
        # Table Title
        scen_title = "SSP5-8.5" if scenario == "ssp585" else "SSP2-4.5"
        table_id = "S1a" if scenario == "ssp585" else "S1b"
        title = f"Table {table_id}: Model GWL Crossing Years and Storyline Assignments ({scen_title})"
        
        # Save files
        df.to_csv(csv_filename, index=False)
        save_dataframe_as_pdf(df, pdf_filename, title)
        save_dataframe_as_png(df, png_filename, title)
        
        print(f"Success! {scenario.upper()} Table created as PDF and PNG.")
        print(f"Total Models ({scenario}): {len(df)}")

if __name__ == "__main__":
    main()
