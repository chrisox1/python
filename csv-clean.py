import os
import pandas as pd
import re

def format_number(val):
    try:
        if pd.isna(val):
            return None
        val_str = str(val).strip()
        val_str = val_str.replace(',', '')
        return float(val_str)
    except Exception:
        return None

def shorten_name(name, existing):
    # Remove extension and keep only alphanumeric characters and underscores
    base = os.path.splitext(name)[0]
    base = re.sub(r'\W+', '_', base)  # Replace non-word characters with underscore
    short = base[:10]  # Limit to 10 chars
    i = 1
    while short in existing:
        short = f"{base[:8]}{i}"  # Avoid duplicates
        i += 1
    return short

def load_and_clean(filepath, short_name):
    try:
        df = pd.read_csv(filepath, usecols=['Date', 'Price'])
    except Exception as e:
        print(f"❌ Error reading '{filepath}': {e}")
        return None

    df['Price'] = df['Price'].apply(format_number)

    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%Y", errors='coerce')

    df = df.dropna(subset=['Date', 'Price'])

    df = df.sort_values('Date')

    df = df.rename(columns={'Price': short_name})

    return df[['Date', short_name]]

def main():
    folder = os.getcwd()
    csv_files = [f for f in os.listdir(folder) if f.lower().endswith('.csv')]

    if not csv_files:
        print("No CSV files found.")
        return

    print("\nAvailable CSV files:")
    for i, file in enumerate(csv_files[:10]):
        print(f"{i+1}: {file}")

    selection = input("\nEnter numbers of CSVs to clean (comma-separated, max 10) or 'x' for all: ").strip().lower()

    if selection == 'x':
        selected_files = csv_files[:10]
    else:
        selected_indices = [int(i.strip()) - 1 for i in selection.split(',') if i.strip().isdigit()][:10]
        selected_files = [csv_files[i] for i in selected_indices if 0 <= i < len(csv_files)]


    if not selected_files:
        print("No valid files selected.")
        return

    dfs = []
    used_short_names = set()
    short_names = []

    for file in selected_files:
        short = shorten_name(file, used_short_names)
        used_short_names.add(short)
        short_names.append(short)

        df = load_and_clean(os.path.join(folder, file), short)
        if df is None or df.empty:
            print(f"⚠️ Skipping '{file}' due to errors or empty data.")
            continue

        dfs.append(df)

    if len(dfs) < 2:
        print("❌ Need at least two valid CSVs to proceed.")
        return

    common_dates = set(dfs[0]['Date'])
    for df in dfs[1:]:
        common_dates = common_dates & set(df['Date'])

    if not common_dates:
        print("❌ No shared dates found across all selected files.")
        return

    for i in range(len(dfs)):
        dfs[i] = dfs[i][dfs[i]['Date'].isin(common_dates)]

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Date')

    merged_df = merged_df.sort_values('Date')

    name_part = '_'.join(short_names)
    output_name = f"cleaned_{name_part}.xlsx"
    merged_df.to_excel(output_name, index=False)

    print(f"\n✅ Final cleaned file saved as '{output_name}' with {len(merged_df)} shared dates.")

if __name__ == "__main__":
    main()
