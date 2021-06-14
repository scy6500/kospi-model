import pandas as pd
import json

def main():
    raw_data = pd.read_csv("data/raw.csv")
    minmax = raw_data.copy()
    std = raw_data.copy()
    minmax_json = dict()
    std_json = dict()

    for feature_name in raw_data.columns:
        if feature_name == "Date":
            continue
        else:
            max_value = float(raw_data[feature_name].max())
            min_value = float(raw_data[feature_name].min())
            minmax[feature_name] = (raw_data[feature_name] - min_value) / (max_value - min_value)
            minmax_json[feature_name + "_max"] = max_value
            minmax_json[feature_name + "_min"] = min_value

            average = float(raw_data[feature_name].mean())
            std_value = float(raw_data[feature_name].std())
            std[feature_name] = (raw_data[feature_name] - average) / std_value
            std_json[feature_name + "_average"] = average
            std_json[feature_name + "_std"] = std_value

    minmax.to_csv("data/minmax.csv", mode='w', index=False)
    std.to_csv("data/std.csv", mode='w', index=False)

    with open('data/minmax.json', 'w') as f:
        json.dump(minmax_json, f, indent="\t")
    f.close()

    with open('data/std.json', 'w') as f:
        json.dump(std_json, f, indent="\t")
    f.close()