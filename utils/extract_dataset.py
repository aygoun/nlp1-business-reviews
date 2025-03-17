import os


# Function to create a smaller dataset for development
def create_small_dataset(input_file, output_file, num_lines=10000):
    if os.path.exists(output_file):
        print(f"Small dataset already exists at {output_file}")
    else:
        print(f"Creating small dataset with {num_lines} lines...")
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for i, line in enumerate(infile):
                if i >= num_lines:
                    break
                outfile.write(line)
        print(f"Small dataset created at {output_file}")


create_small_dataset(
    "./data_set/yelp_academic_dataset_review.json",
    "./data_set/processed_reviews_small.json",
)
