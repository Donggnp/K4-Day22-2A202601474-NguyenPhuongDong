import os

# 1. Fix 03_dpo_train.py LR
nb3 = "notebooks/03_dpo_train.py"
with open(nb3, "r", encoding="utf-8") as f:
    c3 = f.read()
c3 = c3.replace('"5e-7"', '"5e-5"')
with open(nb3, "w", encoding="utf-8") as f:
    f.write(c3)

# 2. Fix 04_compare_and_eval.py Adapter Stacking
nb4 = "notebooks/04_compare_and_eval.py"
with open(nb4, "r", encoding="utf-8") as f:
    c4 = f.read()
# Replace the loading logic
old_load = "    model = PeftModel.from_pretrained(model, str(adapter_path))"
new_load = """    if "dpo" in str(adapter_path):
        model = PeftModel.from_pretrained(model, str(SFT_PATH))
        model.load_adapter(str(adapter_path), adapter_name="dpo")
        model.set_adapter("dpo")
    else:
        model = PeftModel.from_pretrained(model, str(adapter_path))"""
c4 = c4.replace(old_load, new_load)
with open(nb4, "w", encoding="utf-8") as f:
    f.write(c4)

# 3. Fix 05_merge_deploy_gguf.py Adapter Stacking
nb5 = "notebooks/05_merge_deploy_gguf.py"
if os.path.exists(nb5):
    with open(nb5, "r", encoding="utf-8") as f:
        c5 = f.read()
    old_load5 = "model = PeftModel.from_pretrained(model, str(SFT_PATH))"
    new_load5 = """model = PeftModel.from_pretrained(model, str(SFT_PATH))
model.load_adapter(str(DPO_PATH), adapter_name="dpo")
model.set_adapter("dpo")"""
    c5 = c5.replace(old_load5, new_load5)
    with open(nb5, "w", encoding="utf-8") as f:
        f.write(c5)

# 4. Fix scripts/merge_and_gguf.py Adapter Stacking
sc5 = "scripts/merge_and_gguf.py"
if os.path.exists(sc5):
    with open(sc5, "r", encoding="utf-8") as f:
        cs5 = f.read()
    old_load_sc5 = "model = PeftModel.from_pretrained(model, args.sft_path)"
    new_load_sc5 = """model = PeftModel.from_pretrained(model, args.sft_path)
    model.load_adapter(args.dpo_path, adapter_name="dpo")
    model.set_adapter("dpo")"""
    cs5 = cs5.replace(old_load_sc5, new_load_sc5)
    with open(sc5, "w", encoding="utf-8") as f:
        f.write(cs5)
