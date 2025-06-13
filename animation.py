import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
import matplotlib.font_manager as fm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm





# Editable source text and font configuration
# Ask user for custom title and source text
# === USER INPUTS ===
chart_title = input("Enter the chart title: ")
SOURCE_TEXT = input("Enter the source text (e.g. 'Source: Coinbase, Investing.com'): ")
video_filename = input("Enter the output video file name (without extension): ") + ".mp4"
fps = int(input("Enter frames per second (e.g. 30): "))


font_path = "font.ttf"  # Ensure this points to your custom font file
custom_font = fm.FontProperties(fname=font_path)

# === CONFIGURATION FOR MULTIPLE DATASETS ===
# Step 1: Scan current directory for Excel files
excel_files = [f for f in os.listdir() if f.endswith((".xls", ".xlsx"))]

if not excel_files:
    raise FileNotFoundError("No Excel files found in the current directory.")

# Step 2: Ask user to pick a file
print("Available Excel files:")
for i, fname in enumerate(excel_files):
    print(f"{i + 1}: {fname}")

choice = int(input("Enter the number of the Excel file you want to use: ")) - 1
selected_file = excel_files[choice]
print(f"\nYou selected: {selected_file}")

# Step 3: Load the file and get columns
df = pd.read_excel(selected_file)
columns = [col for col in df.columns if col.lower() != "date"]

# Optional color palette (repeats if more columns than colors)
color_palette = ["#FF9900", "#145da0", "#FF0000", "#008000", "#800080", "#00CED1", "#DC143C"]

# Step 4: Ask for label for each column
datasets_info = []
for i, col in enumerate(columns):
    label = input(f"Enter label for column '{col}': ")
    datasets_info.append({
        "file_path": selected_file,
        "column_name": col,
        "label": label,
        "color": color_palette[i % len(color_palette)]
    })

# === Output the generated dataset info ===
print("\nGenerated datasets_info:\n")
for d in datasets_info:
    print(d)

# Animation configuration
interval = 1000 // fps
video_path = video_filename
extra_frames = 60            # Extra frames to hold the final frame

# === LOAD AND PREPROCESS EACH DATASET ===
all_datasets = []

for ds in datasets_info:
    file_path = ds["file_path"]
    if file_path.endswith(".xlsx"):
        df = pd.read_excel(file_path)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format for {file_path}. Use .xlsx or .csv")
    
    column_name = ds["column_name"]
    if "Date" not in df.columns or column_name not in df.columns:
        raise ValueError(f"Missing required columns in {file_path}: 'Date' or '{column_name}'")
    
    df = df.dropna(subset=[column_name])
    df = df[df[column_name].apply(lambda x: np.isfinite(x))]
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df = df[df['Date'] <= pd.Timestamp.today()]  # Remove future dates

    values = df[column_name].values
    dates = df['Date'].values

    # Instead of percent growth, compute normalized values (price divided by the first price)
    normalized_values = values / values[0]
    # Apply slight Gaussian smoothing if desired
    normalized_values_smoothed = gaussian_filter1d(normalized_values, sigma=0.8)
    
    x_values = np.arange(len(values))
    
    print(f"Dataset '{ds['label']}': Base price = {values[0]}, "
          f"Normalized min = {normalized_values.min():.2f}, max = {normalized_values.max():.2f}")
    
    all_datasets.append({
        "x_values": x_values,
        "raw_values": values,
        "normalized_values": normalized_values,
        "normalized_values_smoothed": normalized_values_smoothed,
        "dates": dates,
        "label": ds["label"],
        "color": ds["color"]
    })

max_frames = max(len(ds["x_values"]) for ds in all_datasets)
total_frames = max_frames + extra_frames

# === SET UP FIGURE, AXES, AND GLOBAL STYLE ===
plt.rcParams["image.resample"] = False
fig, ax = plt.subplots(figsize=(12, 6))
plt.subplots_adjust(left=0.08, right=0.92, top=0.90, bottom=0.10)

# Set initial y–axis limits (for normalized prices, starting at 1)
ax.set_ylim(0.5, 10)
ax.yaxis.set_label_position("right")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{(y - 1) * 100:.0f}%"))
ax.yaxis.tick_right()
ax.set_ylabel("Percent Moved Since Start)", fontsize=9, color="#00FA9A", labelpad=15, fontproperties=custom_font)
ax.set_title(chart_title, color="#FF4500", fontproperties=custom_font)
ax.title.set_fontsize(45)

# Use a logarithmic scale on the y–axis
# Use a logarithmic scale on the y–axis and remove the default log tick labels (like 10^x)




# Set dark background and styling
fig.patch.set_facecolor("#121212")
ax.set_facecolor("#1E1E1E")
ax.spines['bottom'].set_color('#FFD700')
ax.spines['right'].set_color('#FFD700')
ax.yaxis.label.set_color('#FFD700')
ax.tick_params(axis='y', colors='#FFD700', labelright=True, right=True, left=False)
ax.xaxis.set_visible(False)
ax.spines['left'].set_visible(False)

# === LOAD AND ADD THE LOGO (OPTIONAL) ===
logo_path = "nicdfdfe.png"  # Change to your logo path
if os.path.exists(logo_path):
    logo = plt.imread(logo_path)
    logo_size = 0.025  # Adjust scaling factor as needed
    imagebox = OffsetImage(logo, zoom=logo_size, alpha=1.0)
    ab = AnnotationBbox(imagebox, (1, 1), xycoords='axes fraction',
                        frameon=False, box_alignment=(1, 1))
    ax.add_artist(ab)
else:
    print("Warning: Logo not found, skipping.")

# === INITIALIZE PLOT ELEMENTS FOR EACH DATASET ===
lines = []  # One line per dataset
for ds in all_datasets:
    line, = ax.plot([], [], lw=3, color=ds["color"], linestyle='solid')
    lines.append(line)

# Create dynamic text annotations for each dataset.
dataset_annotations = []
for ds in all_datasets:
    annot = ax.text(0, 0, ds["label"], fontsize=18, color=ds["color"],
                    fontproperties=custom_font, weight='bold', va='center')
    dataset_annotations.append(annot)

# Combined annotation box for date and price info.
text_box = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=12,
                   color="white", verticalalignment='top', horizontalalignment='left',
                   fontproperties=custom_font)

# Static text at the bottom.
fig.text(0.02, 0.02, "@TheBitcoinRun", fontsize=18, fontweight='bold', fontstyle='italic',
         color="white", ha="left", va="bottom", fontproperties=custom_font)
fig.text(0.98, 0.02, SOURCE_TEXT, fontsize=18, fontstyle='italic',
         color="white", ha="right", va="bottom", fontproperties=custom_font)
fig.text(0.5, 0.02, "", fontsize=18, fontweight='bold',
         color="white", ha="center", va="bottom", fontproperties=custom_font)

# === ANIMATION FUNCTIONS ===
def init():
    for line in lines:
        line.set_data([], [])
    text_box.set_text("")
    for annot in dataset_annotations:
        annot.set_position((0, 0))
    return lines + dataset_annotations + [text_box]

def update(frame):
    global ani
    # Slow down the final frames for a smoother finish.
    slowdown_start = max_frames - 30
    if frame >= slowdown_start:
        slowdown_factor = 1 + ((frame - slowdown_start) / 10)
    else:
        slowdown_factor = 1
    ani.event_source.interval = int(interval * slowdown_factor)
    
    x_min_candidates = []
    x_max_candidates = []
    y_min_candidates = []
    y_max_candidates = []
    
    price_info = ""

    for i, ds in enumerate(all_datasets):
        ds_length = len(ds["x_values"])
        if ds_length == 0:
            continue
        current_index = frame if frame < ds_length else ds_length - 1
        
        current_indices = ds["x_values"][:current_index+1]
        # Use the normalized (and smoothed) values for plotting
        current_norm = ds["normalized_values_smoothed"][:current_index+1]
        lines[i].set_data(current_indices, current_norm)
        
        if len(current_indices) > 0:
            x_min_candidates.append(current_indices[0])
            x_max_candidates.append(current_indices[-1])
            y_min_candidates.append(np.min(current_norm))
            y_max_candidates.append(np.max(current_norm))
            
            start_price = ds["raw_values"][0]
            current_price = ds["raw_values"][current_index]
            percent_change = ((current_price - start_price) / start_price) * 100
            price_info += f"{ds['label']}: {percent_change:+.2f}%\n"

            
            x_last = current_indices[-1]
            y_last = current_norm[-1]
            dataset_annotations[i].set_position((x_last, y_last))
    
    if all_datasets:
        ds0 = all_datasets[0]
        ds0_length = len(ds0["x_values"])
        current_index0 = frame if frame < ds0_length else ds0_length - 1
        current_date_str = pd.to_datetime(ds0["dates"][current_index0]).strftime('%Y-%m-%d')
        combined_annotation = f"Date: {current_date_str}\n" + price_info
    else:
        combined_annotation = price_info

    if x_min_candidates and x_max_candidates:
        ax.set_xlim(min(x_min_candidates), max(x_max_candidates))
    if y_min_candidates and y_max_candidates:
        margin = 0.1
        new_min = min(y_min_candidates) / (1 + margin)
        new_max = max(y_max_candidates) * (1 + margin)
        ax.set_ylim(new_min, new_max)
    
    text_box.set_text(combined_annotation)

    # (Optional: Add vertical lines or other annotations here)
    return lines + dataset_annotations + [text_box]

# Style the y–tick labels using the custom font.
for label in ax.get_yticklabels():
    label.set_fontproperties(custom_font)
    label.set_fontsize(14)
    label.set_color("#FFD700")

# === CREATE AND SAVE THE ANIMATION ===
# Create a generator with tqdm for progress bar
def frame_generator():
    for frame in tqdm(range(total_frames), desc="Rendering animation"):
        yield frame

ani = animation.FuncAnimation(fig, update, frames=frame_generator(), init_func=init,
                              blit=False, interval=interval)


video_path = video_filename
writer = animation.FFMpegWriter(fps=fps, extra_args=["-preset", "ultrafast"])
ani.save(video_path, writer=writer, dpi=150)

print("\nAnimation saved successfully! ✅")
