import json

# Input and output file paths
input_file = "line_charts_processed_visual_linguistic_train_randomized_with_llm_as_a_judge_400.json"
output_file = "line_charts_processed_visual_linguistic_train_randomized_with_llm_as_a_judge_400_prefix_fixed.json"

# Prefixes to replace
old_prefix = "/home/ramvenkat98/"
new_prefix = "/nlp/scr/ram1998/"

# Read the JSON file
with open(input_file, "r") as f:
    data = json.load(f)

# Update the image paths
for item in data:
    item["image"] = [path.replace(old_prefix, new_prefix) for path in item["image"]]

# Write the updated data back to a new JSON file
with open(output_file, "w") as f:
    json.dump(data, f, indent=4)

print(f"Updated JSON file has been saved to {output_file}")

