import glob

directory = './data/knowledge'  # 要读取文件的目录

files = glob.glob(directory + '/*')  # 获取目录中的所有文件，并存放在一个列表中

content = []
for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content.append(f.read())

with open('content.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n----------------------------------------\n\n'.join(content))