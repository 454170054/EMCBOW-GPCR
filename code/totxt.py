import pandas as pd


def get_text(path):
    path = str(path)
    data = pd.read_excel(path)
    file_name = path.replace("all_data.xlsx", "all_5.txt")
    print(file_name)
    with open(file_name, "w") as fr:
        for i in data.seq:
            text = ""
            for j in range(len(i) - 4):
                if j != len(i) - 5:
                    text = text + i[j: j+5] + " "
                else:
                    text = text + i[j: j+5]
            fr.write(text + "\n")


if __name__ == '__main__':
    path = r'../files/all_data.xlsx'
    get_text(path)