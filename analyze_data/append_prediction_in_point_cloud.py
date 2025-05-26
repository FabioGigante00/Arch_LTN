import os

# Set directories

print("Current working directory:", os.getcwd())
txt_dir = '../datasets/Arch_Feat/Test_splitted_CC'
pred_dir = './results'
out_dir = './results/pred_appended'

# List all .txt files in txt_dir (excluding _pred.txt)
all_txt_files = os.listdir(txt_dir)
txt_files = [f for f in all_txt_files if f.endswith('.txt') and not f.endswith('_pred.txt')]

# Only keep those with a matching _pred.txt in pred_dir
all_pred_files = os.listdir(pred_dir)
matching_files = [f for f in txt_files if f[:-4] + '_pred.txt' in all_pred_files]


print(f"Found {len(matching_files)} matching files.")


for txt_file in matching_files:
    pred_file = txt_file[:-4] + '_pred.txt'
    txt_path = os.path.join(txt_dir, txt_file)
    pred_path = os.path.join(pred_dir, pred_file)
    output_lines = []
    count_9_points = 0

    with open(txt_path, 'r') as f_txt, open(pred_path, 'r') as f_pred:
        txt_lines = f_txt.readlines()
        pred_lines = f_pred.readlines()
        # Assert exact same len and print column shape
        assert len(txt_lines) == len(pred_lines), f"Line count mismatch in {txt_file} and {pred_file}"
        print(f"{txt_file}: {len(txt_lines)} lines, columns: {len(txt_lines[0].strip().split())}")
        print(f"{pred_file}: {len(pred_lines)} lines, columns: {len(pred_lines[0].strip().split())}")
        for i, (txt_line, pred_line) in enumerate(zip(txt_lines, pred_lines)):
            columns = txt_line.strip().split()
            pred_columns = pred_line.strip().split()
            if len(columns) >= 10 and len(pred_columns) > 0:
                if columns[9] != '9.000000':
                    new_value = pred_columns[-1]
                else:
                    new_value = '9.000000'
                    count_9_points += 1
                new_line = txt_line.strip() + ' ' + new_value + '\n'
            else:
                new_line = txt_line  # Leave unchanged if not enough columns
            output_lines.append(new_line)

    print(f"{txt_file}: {count_9_points} points with value 9.000000")

    # Write to a new file in out_dir with _appended.txt suffix
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, txt_file[:-4] + '_appended.txt')
    with open(output_path, 'w') as f_out:
        f_out.writelines(output_lines)