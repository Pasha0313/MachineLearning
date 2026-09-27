import os
def load_config(path="model_config.txt"):
    config = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    config[key.strip()] = value.strip()
    return config

def get_control_settings(config):
    model_name = config.get("model_name", "distilgpt2")  #gpt2-medium
    auto_run = config.get("auto_run", "False").lower() == "true"
    live_output = config.get("live_output", "True").lower() == "true"
    user_input = config.get("user_input", "5,6,7,8")

    print(f"🤖 Model selected: {model_name}")
    print(f"⚙️ AUTO_RUN: {auto_run}, Live Output: {live_output}, Steps: {user_input}")
    return model_name, auto_run, live_output, user_input