import pandas as pd
import json
from pathlib import Path

def csv_to_label_studio_json(csv_path, output_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Group the dataframe by run_id and frame_id
    grouped = df.groupby(['run_id', 'frame_id'])

    # Initialize the results list
    results = []

    # Iterate through the grouped data
    for (run_id, frame_id), group in grouped:
        # Create a new annotation for each group
        annotation = {
            "id": f"{run_id}_{frame_id}",
            "result": [
                {
                    "id": f"result_{run_id}_{frame_id}",
                    "type": "rectangle",
                    "from_name": "bbox",
                    "to_name": "video",
                    "original_width": 1920,
                    "original_height": 1080,
                    "image_rotation": 0,
                    "value": {
                        "x": group['x0'].iloc[0],
                        "y": group['y0'].iloc[0],
                        "width": (group['x1'].iloc[0] - group['x0'].iloc[0]),
                        "height": (group['y1'].iloc[0] - group['y0'].iloc[0]),
                        "rotation": 0,
                        "labels": [str(run_id)]
                    },
                    "frame": int(frame_id),
                    "meta": {
                        "angle": group['angle'].iloc[0]
                    }
                }
            ]
        }
        results.append(annotation)

    # Create the final JSON structure
    output_json = {
        "data": {
            "video_url": "your_video_url_here"
        },
        "annotations": [
            {
                "result": results
            }
        ]
    }

    # Write the JSON to a file
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)

    print(f"Label Studio JSON file created at: {output_path}")

# Example usage
csv_path = ""
output_path = ""
csv_to_label_studio_json(csv_path, output_path)