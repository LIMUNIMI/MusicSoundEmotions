import re
import pandas as pd
from collections import defaultdict

# Define regex patterns to capture the relevant parts of the log lines
title_pattern = re.compile(r"^([A-Z][A-Z\s\-\(\)\+,\?emo]+)\?$")
ratio_pattern = re.compile(r"Ratio:\s(.+?)\s\+\s(.+?)$")
model_pattern = re.compile(r"Tuning\s(\S+)")
metrics_pattern = re.compile(r"Obtained metrics for (\S+)")
r2_pattern = re.compile(r"r2,\sRMSE,\sMAE")
values_pattern = re.compile(r"(-?\d+\.\d+e[+-]?\d+)\s±\s(\d+\.\d+e[+-]?\d+)")

data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))


# Function to parse log file
def parse_log_file(file_path):
    current_title = None
    current_ratio = None
    current_model = None
    current_set = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            title_match = title_pattern.match(line)
            ratio_match = ratio_pattern.search(line)
            model_match = model_pattern.search(line)
            metrics_match = metrics_pattern.search(line)
            r2_match = r2_pattern.match(line)
            values_match = values_pattern.findall(line)

            if title_match:
                current_title = title_match.group(1).strip()
                print("New title:", current_title)
                current_ratio = None
                current_model = None
                current_set = None
                continue

            if ratio_match:
                current_ratio = f"{ratio_match.group(1)} + {ratio_match.group(2)}"
                print("\tNew ratio:", current_ratio)
                continue

            if model_match:
                current_model = model_match.group(1).strip()
                print("\t\tNew model:", current_model)
                continue

            if metrics_match:
                current_set = metrics_match.group(1).strip()
                print("\t\t\tNew set:", current_set)
                continue

            if len(values_match) == 0:
                # skip non-matching lines
                continue

            print("\t\t\t\tNew metrics:", values_match)
            if (
                r2_match
                and current_title
                and current_ratio
                and current_model
                and current_set
            ):
                # Overwrite with new data if it already exists
                data[current_title][current_ratio][current_model][current_set] = []
                continue

            if (
                values_match
                and current_title
                and current_ratio
                and current_model
                and current_set
            ):
                data[current_title][current_ratio][current_model][current_set].append(
                    values_match[0]
                )


# Function to save parsed data into multiple CSV files
def save_to_csv(data):
    for title, ratios in data.items():
        rows = []
        for ratio, models in ratios.items():
            row = {"Ratio": ratio}
            for model, sets in models.items():
                for set_type, values in sets.items():
                    if values == []:
                        row[f"{model} {set_type}"] = "errored"
                    else:
                        r2_value = values[0][0] if values != [] else "N/A"
                        r2_error = values[0][1] if values != [] else "N/A"
                        rmse_value = values[1][0] if values != [] else "N/A"
                        rmse_error = values[1][1] if values != [] else "N/A"
                        mae_value = values[2][0] if values != [] else "N/A"
                        mae_error = values[2][1] if values != [] else "N/A"

                        row[f"{model} {set_type} r2_value"] = r2_value
                        row[f"{model} {set_type} r2_error"] = r2_error
                        row[f"{model} {set_type} rmse_value"] = rmse_value
                        row[f"{model} {set_type} rmse_error"] = rmse_error
                        row[f"{model} {set_type} mae_value"] = mae_value
                        row[f"{model} {set_type} mae_error"] = mae_error
            rows.append(row)

        df = pd.DataFrame(rows)
        print(f"Saving {title}...")
        output_filename = f"{title.replace(' ', '_')}.csv"
        df.to_csv(output_filename, index=False)


# Main function to execute the script
def main(*log_files):
    """
    Parses log files and saves the parsed data into multiple CSV files.
    """
    for log_file in log_files:
        print(f"Parsing {log_file}...")
        parse_log_file(log_file)

    save_to_csv(data)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
