 
import os
from datetime import datetime
import matplotlib.pyplot as plt

# Define the directory to save results
results_dir = os.path.join(os.getcwd(), 'public/results')

# Ensure the directory exists
os.makedirs(results_dir, exist_ok=True)

# Generate a text result
text_result = "Result generated at " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
text_file_path = os.path.join(results_dir, 'result.txt')

# Save the text result
with open(text_file_path, 'w') as f:
    f.write(text_result)

# Generate a simple plot as an image result
plt.figure()
plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.title("Sample Plot")
image_file_path = os.path.join(results_dir, 'result.png')

# Save the plot
plt.savefig(image_file_path)
plt.close()
 