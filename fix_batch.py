import os
import re

for nb_dir in ["notebooks", "scripts", "colab"]:
    if not os.path.exists(nb_dir): continue
    for nb in os.listdir(nb_dir):
        if nb.endswith(".py") or nb.endswith(".ipynb"):
            filepath = os.path.join(nb_dir, nb)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            content = re.sub(r'PER_DEVICE_BATCH\s*=\s*\d+', 'PER_DEVICE_BATCH = 2', content)
            content = re.sub(r'GRAD_ACCUM\s*=\s*\d+', 'GRAD_ACCUM = 4', content)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
