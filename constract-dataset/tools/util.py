import yaml


def save_yaml(data: list, save_path: str):
    with open(save_path, "w+", encoding="utf-8") as f:
        yaml.dump(data, f)


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data
