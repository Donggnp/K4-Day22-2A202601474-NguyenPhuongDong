import os

chatml_code = "tokenizer.chat_template = \"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n{% endif %}<|im_start|>{{ message['role'] }}\\n{{ message['content'] }}<|im_end|>\\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\\n{% endif %}\""

for nb_dir in ["notebooks", "scripts", "colab"]:
    if not os.path.exists(nb_dir): continue
    for nb in os.listdir(nb_dir):
        if nb.endswith(".py"):
            filepath = os.path.join(nb_dir, nb)
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            with open(filepath, "w", encoding="utf-8") as f:
                for line in lines:
                    # Xóa toàn bộ các dòng chèn chat_template cũ bị lỗi (do nằm sai chỗ hoặc thụt lề sai)
                    if "tokenizer.chat_template =" in line and "{% for message" in line:
                        continue
                    
                    # Chèn chat_template MỚI ngay TRƯỚC dòng if tokenizer... để đảm bảo nó luôn chạy và giữ đúng khoảng trắng
                    if "if tokenizer.pad_token is None:" in line:
                        leading = line[:len(line) - len(line.lstrip())]
                        f.write(leading + chatml_code + "\n")
                    
                    f.write(line)

