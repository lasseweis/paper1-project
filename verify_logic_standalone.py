
import glob
import os
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class MockConfig:
    CMIP6_RAW_PR_PATH_PATTERN = '/data/reloclim/normal/CMIP6_STREAM/paper1-cmip-data/pr_regrid/pr_Amon_{model}_{scenario}_*_*_regridded.nc'

class MockAnalyzer:
    def __init__(self):
        self.config = MockConfig()

    def find_cmip6_pr_file(self, model, scenario):
        """
        Searches for the raw PR (precipitation) file for a given model and scenario.
        Handles cases where both standard (YYYYMM-YYYYMM) and non-standard (e.g. YYYYMMDD-YYYYMMDD) files exist,
        prioritizing standard files and filling gaps with non-standard ones to avoid overlaps.
        """
        search_pattern = self.config.CMIP6_RAW_PR_PATH_PATTERN.format(model=model, scenario=scenario)
        found_files = glob.glob(search_pattern)
        
        if not found_files:
            return []
            
        member_groups = {}
        for f in found_files:
            parent_dir = os.path.dirname(f)
            if parent_dir not in member_groups:
                member_groups[parent_dir] = []
            member_groups[parent_dir].append(f)
            
        if not member_groups: return []
            
        sorted_dirs = sorted(member_groups.keys())
        target_dir = sorted_dirs[0]
        
        member_files = sorted(member_groups[target_dir])
        
        # --- Smart Filtering Logic ---
        standard_files = []
        non_standard_files = []
        
        # Regex for YYYYMM-YYYYMM (Standard)
        re_standard = re.compile(r'_(\d{6})-(\d{6})_')
        
        # Regex for YYYYMMDD-YYYYMMDD (Non-Standard)
        re_non_std = re.compile(r'_(\d{8})-(\d{8})_')

        for f in member_files:
            basename = os.path.basename(f)
            match_std = re_standard.search(basename)
            match_non = re_non_std.search(basename)
            
            if match_std:
                start_str, end_str = match_std.groups()
                s_code = int(start_str)
                e_code = int(end_str)
                standard_files.append({
                    'path': f,
                    'start': s_code,
                    'end': e_code,
                    'type': 'standard'
                })
            elif match_non:
                start_str, end_str = match_non.groups()
                s_code = int(start_str[:6])
                e_code = int(end_str[:6])
                non_standard_files.append({
                    'path': f,
                    'start': s_code,
                    'end': e_code,
                    'type': 'non_standard'
                })
            else:
                logging.warning(f"File {basename} does not match expected date pattern, including it in standard set.")
                standard_files.append({
                    'path': f,
                    'start': 0, 
                    'end': 999999,
                    'type': 'unknown'
                })

        # 1. Start with all Standard files
        final_files = [item['path'] for item in standard_files]
        
        #Helper to check overlap against accepted ranges
        def overlaps_with_accepted(candidate_start, candidate_end, accepted_list):
            for item in accepted_list:
                if (candidate_start <= item['end']) and (candidate_end >= item['start']):
                    return True
            return False

        # 2. Add Non-Standard files ONLY if they don't overlap with existing accepted files
        for ns in non_standard_files:
            if not overlaps_with_accepted(ns['start'], ns['end'], standard_files):
                final_files.append(ns['path'])
                logging.info(f"Including non-standard file to fill gap: {os.path.basename(ns['path'])}")
            else:
                logging.warning(f"Excluding overlapping non-standard file: {os.path.basename(ns['path'])}")
        
        return sorted(final_files)

def run_test():
    analyzer = MockAnalyzer()
    model = "EC-Earth3-Veg-LR"
    scenario = "ssp585"
    
    print(f"Testing logic for {model} {scenario}...")
    files = analyzer.find_cmip6_pr_file(model, scenario)
    
    print(f"\nResulting file list ({len(files)} files):")
    
    has_standard = False
    has_rogue = False
    
    for f in files:
        basename = os.path.basename(f)
        # print(f" - {basename}")
        if "204201-204212" in basename:
            has_standard = True
        if "20420116-21001216" in basename:
            has_rogue = True
            
    if has_standard and not has_rogue:
        print("\nSUCCESS: Standard file present, Rogue file excluded.")
    elif has_rogue:
        print("\nFAILURE: Rogue file is still present!")
    else:
        print("\nFAILURE: Standard file missing?")

if __name__ == "__main__":
    run_test()
