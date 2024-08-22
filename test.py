import os
from PIL import Image
import json
def read_json(path):
    with open(path,'r') as f:
        data = json.load(f)
    return data
captions = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/pmcimage_captions_short.json")

print(len(os.listdir("/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_0")))
print(len(os.listdir("/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_1")))
print(os.listdir("/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_1")[-2])
path = os.path.join("/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_1",os.listdir("/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_1")[-2])
image = Image.open(path)
image.save("./test.jpg")
idx = int(os.listdir("/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_1")[-2].split(".")[0])
print(idx)
print(captions[idx])