"""
Search for dates with both Sentinel-2 and meaningful CyaN data.

This script finds the next available date starting from a given start date where:
1. A Sentinel-2 product is available for the bounding box
2. The corresponding CyaN data contains meaningful pixels (not just 0, 254, 255)

The script will display the CyaN data when a suitable date is found.
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from constants import cyan_colormap

# Add the case_studies directory to path to import functions
sys.path.append(os.path.join(os.path.dirname(__file__), 'case_studies'))
from get_cyan import search_sentinel2, download_cyan_geotiff, crop_cyan_to_bbox, get_cyan_url

# USER CONFIGURATION
BOUNDING_BOX = [
    -85.972191,
    42.472820,
    -85.941217,
    42.454071,
]  # Swan Lake [min_lon, max_lat, max_lon, min_lat]

CYAN_REGION_ID = "6_2"  # CyaN region ID
START_DATE = datetime(2023, 7, 1)  # Starting date to search from
MAX_DAYS_TO_SEARCH = 60  # Maximum number of days to search
CYAN_CACHE_DIR = "./cyan_search_cache"  # Directory to cache CyaN downloads

def check_cyan_meaningful_data(cyan_array, min_meaningful_ratio=0.01):
    """
    Check if CyaN data contains meaningful pixels (not just no-data values).
    
    Args:
        cyan_array: numpy array of CyaN data
        min_meaningful_ratio: minimum ratio of meaningful pixels required
        
    Returns:
        tuple: (has_meaningful_data: bool, meaningful_ratio: float, stats: dict)
    """
    total_pixels = cyan_array.size
    
    # Count different pixel types
    zero_pixels = np.sum(cyan_array == 0)
    nodata_254 = np.sum(cyan_array == 254) 
    nodata_255 = np.sum(cyan_array == 255)
    meaningful_pixels = total_pixels - zero_pixels - nodata_254 - nodata_255
    
    meaningful_ratio = meaningful_pixels / total_pixels
    
    stats = {
        'total_pixels': total_pixels,
        'zero_pixels': zero_pixels,
        'nodata_254': nodata_254,
        'nodata_255': nodata_255,
        'meaningful_pixels': meaningful_pixels,
        'meaningful_ratio': meaningful_ratio
    }
    
    return meaningful_ratio >= min_meaningful_ratio, meaningful_ratio, stats

def display_cyan_data(cyan_array, date, stats):
    """
    Display the CyaN data with statistics.
    
    Args:
        cyan_array: numpy array of CyaN data
        date: datetime object for the date
        stats: statistics dictionary from check_cyan_meaningful_data
    """
    # Apply colormap
    cyan_colored = cyan_colormap[cyan_array]
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: CyaN data with colormap
    im1 = ax1.imshow(cyan_colored)
    ax1.set_title(f'CyaN HAB Data - {date.strftime("%Y-%m-%d")}', fontsize=12)
    ax1.axis('off')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('CyaN Index', rotation=270, labelpad=15)
    
    # Right plot: Raw data histogram
    ax2.hist(cyan_array.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('CyaN Index Value')
    ax2.set_ylabel('Pixel Count')
    ax2.set_title('CyaN Value Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = (
        f"Total pixels: {stats['total_pixels']:,}\n"
        f"Zero pixels: {stats['zero_pixels']:,}\n"
        f"No-data (254): {stats['nodata_254']:,}\n"
        f"No-data (255): {stats['nodata_255']:,}\n"
        f"Meaningful pixels: {stats['meaningful_pixels']:,}\n"
        f"Meaningful ratio: {stats['meaningful_ratio']:.1%}"
    )
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'CyaN Data Analysis - {date.strftime("%Y-%m-%d")}', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return fig

def search_for_suitable_date():
    """
    Search for a date with both Sentinel-2 availability and meaningful CyaN data.
    
    Returns:
        tuple: (found_date, cyan_path, cyan_array) or (None, None, None) if not found
    """
    # Create cache directory
    os.makedirs(CYAN_CACHE_DIR, exist_ok=True)
    
    current_date = START_DATE
    dates_checked = 0
    
    print(f"Starting search from {START_DATE.strftime('%Y-%m-%d')}")
    print(f"Bounding box: {BOUNDING_BOX}")
    print(f"CyaN region: {CYAN_REGION_ID}")
    print(f"Maximum days to search: {MAX_DAYS_TO_SEARCH}")
    print("-" * 50)
    
    while dates_checked < MAX_DAYS_TO_SEARCH:
        try:
            print(f"\nChecking date: {current_date.strftime('%Y-%m-%d')} (Day {dates_checked + 1})")
            
            # Step 1: Check if Sentinel-2 data is available
            print("  Searching for Sentinel-2 data...")
            try:
                sen2_product = search_sentinel2(BOUNDING_BOX, current_date, date_range_days=0)
                product_date = sen2_product["ContentDate"]["Start"][:10]
                print(f"  ✓ Sentinel-2 product available: {product_date}")
                
                # If the found product is not for the exact date, update current_date
                if product_date != current_date.strftime('%Y-%m-%d'):
                    print(f"  → Using Sentinel-2 date: {product_date}")
                    current_date = datetime.strptime(product_date, '%Y-%m-%d')
                
            except (ValueError, ConnectionError) as e:
                print(f"  ✗ No Sentinel-2 data available: {e}")
                current_date += timedelta(days=1)
                dates_checked += 1
                continue
            
            # Step 2: Download and check CyaN data
            print("  Checking CyaN data...")
            
            # Create cache filename for CyaN
            day_of_year = current_date.strftime("%j")
            year = current_date.strftime("%Y")
            cyan_cache_filename = f"L{year}{day_of_year}.L3m_DAY_CYAN_CI_cyano_CYAN_CONUS_300m_{CYAN_REGION_ID}.tif"
            cached_cyan_path = os.path.join(CYAN_CACHE_DIR, cyan_cache_filename)
            
            # Check if we already have this CyaN file cached
            if os.path.exists(cached_cyan_path):
                print(f"  Using cached CyaN file: {cached_cyan_path}")
                original_cyan_path = cached_cyan_path
            else:
                # Download to cache
                print(f"  Downloading CyaN data to cache...")
                try:
                    download_cyan_geotiff(cached_cyan_path, current_date, CYAN_REGION_ID)
                    original_cyan_path = cached_cyan_path
                    print(f"  ✓ Downloaded CyaN data")
                except (ConnectionError, ValueError) as e:
                    print(f"  ✗ Failed to download CyaN data: {e}")
                    current_date += timedelta(days=1) 
                    dates_checked += 1
                    continue
            
            # Step 3: Crop CyaN data to bounding box
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
                cropped_cyan_path = tmp_file.name
            
            try:
                print("  Cropping CyaN data to bounding box...")
                crop_cyan_to_bbox(original_cyan_path, BOUNDING_BOX, cropped_cyan_path)
                print("  ✓ CyaN data cropped successfully")
            except Exception as e:
                print(f"  ✗ Failed to crop CyaN data: {e}")
                current_date += timedelta(days=1)
                dates_checked += 1
                continue
            
            # Step 4: Check if CyaN data contains meaningful pixels
            try:
                img = Image.open(cropped_cyan_path)
                cyan_array = np.array(img)
                
                has_meaningful, meaningful_ratio, stats = check_cyan_meaningful_data(cyan_array)
                
                print(f"  CyaN data stats:")
                print(f"    Total pixels: {stats['total_pixels']:,}")
                print(f"    Meaningful pixels: {stats['meaningful_pixels']:,}")
                print(f"    Meaningful ratio: {meaningful_ratio:.1%}")
                
                if has_meaningful:
                    print(f"  ✓ Found suitable date with meaningful CyaN data!")
                    print(f"  Final date: {current_date.strftime('%Y-%m-%d')}")
                    
                    # Display the data
                    display_cyan_data(cyan_array, current_date, stats)
                    
                    return current_date, cropped_cyan_path, cyan_array
                else:
                    print(f"  ✗ Insufficient meaningful CyaN data ({meaningful_ratio:.1%})")
                    
            except Exception as e:
                print(f"  ✗ Failed to analyze CyaN data: {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(cropped_cyan_path):
                    os.unlink(cropped_cyan_path)
            
            # Move to next date
            current_date += timedelta(days=1)
            dates_checked += 1
            
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            current_date += timedelta(days=1)
            dates_checked += 1
            continue
    
    print(f"\nSearch completed. No suitable date found within {MAX_DAYS_TO_SEARCH} days.")
    return None, None, None

def main():
    """Main function to run the date search."""
    print("=" * 60)
    print("SEARCHING FOR SUITABLE DATES WITH SENTINEL-2 AND CYAN DATA")
    print("=" * 60)
    
    result_date, cyan_path, cyan_array = search_for_suitable_date()
    
    if result_date:
        print(f"\n🎉 SUCCESS!")
        print(f"Found suitable date: {result_date.strftime('%Y-%m-%d')}")
        print(f"CyaN data saved to: {cyan_path}")
        
        # Show unique CyaN values
        unique_values = np.unique(cyan_array)
        print(f"Unique CyaN values in cropped data: {unique_values[:10]}{'...' if len(unique_values) > 10 else ''}")
        
    else:
        print("\n❌ No suitable date found.")
        print("Consider:")
        print("- Increasing MAX_DAYS_TO_SEARCH")
        print("- Trying a different START_DATE")
        print("- Checking a different BOUNDING_BOX")
        print("- Using a different CYAN_REGION_ID")

if __name__ == "__main__":
    main()