import json

def reformat_faq(input_path="Dataset/FAQ.json", output_path="Dataset/FAQ_modified.json"):
    with open(input_path, 'r') as f:
        data = json.load(f)

    new_data = []
    for entry in data:
        new_entry = {
            "question": entry.get("Question", "").strip(),
            "answer": entry.get("Response", "").strip(),
            "metadata": {
                "level": entry.get("Level", "").strip(),
                "school": entry.get("School", "").strip(),
                "program_code": entry.get("Program Code", "").strip(),
                "program": entry.get("Program", "").strip()
            }
        }
        new_data.append(new_entry)

    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

    print(f"✅ Reformatted JSON saved to {output_path}")

if __name__ == "__main__":
    reformat_faq()
