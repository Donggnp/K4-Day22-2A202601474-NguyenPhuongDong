import os
import re

chatml_code = "    tokenizer.chat_template = \"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n{% endif %}<|im_start|>{{ message['role'] }}\\n{{ message['content'] }}<|im_end|>\\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}\""

for nb_dir in ["notebooks", "scripts", "colab"]:
    if not os.path.exists(nb_dir):
        continue
    for nb in os.listdir(nb_dir):
        if nb.endswith(".py") or nb.endswith(".ipynb"):
            filepath = os.path.join(nb_dir, nb)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            content = content.replace("5CD-AI/Vietnamese-alpaca-cleaned", "bkai-foundation-models/vi-alpaca")
            content = content.replace("Qwen/Qwen2.5-1.5B", "unsloth/Qwen2.5-1.5B-bnb-4bit")
            content = re.sub(r'PER_DEVICE_BATCH\s*=\s*\d+', 'PER_DEVICE_BATCH = 8', content)
            content = re.sub(r'GRAD_ACCUM\s*=\s*\d+', 'GRAD_ACCUM = 1', content)
            content = content.replace('use_gradient_checkpointing="unsloth"', 'use_gradient_checkpointing=False')
            content = content.replace('use_gradient_checkpointing=True', 'use_gradient_checkpointing=False')
            
            if "if tokenizer.pad_token is None:" in content and "tokenizer.chat_template =" not in content:
                content = content.replace(
                    "if tokenizer.pad_token is None:\\n",
                    "if tokenizer.pad_token is None:\\n" + chatml_code.replace('"', '\\"') + "\\n"
                )
                content = content.replace(
                    "if tokenizer.pad_token is None:\n",
                    "if tokenizer.pad_token is None:\n" + chatml_code + "\n"
                )

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

