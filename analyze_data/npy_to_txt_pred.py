import os
import numpy as np

print("Current working directory:", os.getcwd())

result_folder = './exp/arch/semseg-pt-v3m1-1_LOCAL/AR-205/result'
out_path = 'results'

if os.path.exists(result_folder) and os.path.isdir(result_folder):
    files = os.listdir(result_folder)
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(result_folder, file)
            arr = np.load(file_path)
            print(f"{file}: shape {arr.shape}")
            print("First three lines:")
            print(arr[:3])
            print("-" * 40)
            # Save as txt in current working directory
            txt_filename = os.path.splitext(file)[0] + '.txt'
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            txt_path = os.path.join(os.getcwd(), out_path, txt_filename)
            np.savetxt(txt_path, arr, fmt='%s')
else:
    print(f"Folder '{result_folder}' does not exist.")