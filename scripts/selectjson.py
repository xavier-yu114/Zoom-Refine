import json
import ijson

input_file = 'path/to/your/dataset.json'  
output_file = 'path/to/your/result.json'  

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    parser = ijson.items(infile, 'item') 

    # Modify the filter condition here
    filtered_data = [
        item for item in parser 
        if (item.get("Subtask") == "Autonomous_Driving") and (item.get("Task") == "Reasoning")
    ]
    
    json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)

print(f"Filtering complete. Results saved to {output_file}")