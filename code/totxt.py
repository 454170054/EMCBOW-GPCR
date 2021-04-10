import pandas as pd
from tqdm import tqdm


def get_text(path):
    for i in range(2, 5):
        path = str(path)
        data = pd.read_excel(path)
        file_name = "all_{}.txt".format(i)
        file_path = path.replace("all_data.xlsx", file_name)
        print(file_path)
        with open(file_path, "w") as fr:
            for j in tqdm(data.seq):
                text = ""
                for k in range(len(j) - i + 1):
                    if k != len(j) - i:
                        text = text + j[k: k+i] + " "
                    else:
                        text = text + j[k: k+i]
                fr.write(text + "\n")


if __name__ == '__main__':
    path = r'../files/all_data.xlsx'
    get_text(path)