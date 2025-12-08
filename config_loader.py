import yaml
from pathlib import Path


def print_config_types(config: dict, prefix: str = ""):
    for key, value in config.items():
        if isinstance(value, dict):
            print_config_types(value, f"{prefix}{key}.")
        else:
            print(f"{prefix}{key}: {type(value).__name__} = {value}")


def deep_update(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_path: str, _visited: set = None, _is_run_config: bool = None
) -> dict:
    if _visited is None:
        _visited = set()

    config_path = Path(config_path).resolve()

    if str(config_path) in _visited:
        raise ValueError(f"Circular dependency detected: {config_path}")

    _visited.add(str(config_path))

    # determine config type from path
    parts = config_path.parts
    if "runs" in parts:
        config_type = "runs"
        if _is_run_config is None:
            _is_run_config = True
    elif "base" in parts:
        config_type = "base"
    else:
        raise ValueError(
            f"Config must be in 'configs/base/' or 'configs/runs/' folder. "
            f"Got: {config_path}"
        )

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "inherit" in config:
        inherit_value = config.pop("inherit")

        if isinstance(inherit_value, str):
            inherit_list = [inherit_value]
        elif isinstance(inherit_value, list):
            inherit_list = inherit_value
        else:
            raise ValueError(
                f"'inherit' must be a string or list, got {type(inherit_value)}"
            )

        merged_base = {}
        for parent_file in inherit_list:
            parent_path = Path(parent_file)
            if parent_path.suffix not in [".yaml", ".yml"]:
                parent_file = f"{parent_file}.yaml"
                parent_path = Path(parent_file)

            if not parent_path.is_absolute() and len(parent_path.parts) == 1:
                configs_root = config_path
                while configs_root.name not in ["configs", "base", "runs"]:
                    configs_root = configs_root.parent
                if configs_root.name in ["base", "runs"]:
                    configs_root = configs_root.parent
                parent_path = configs_root / "base" / parent_file
            else:
                parent_path = config_path.parent / parent_file

            parent_path = parent_path.resolve()

            parent_parts = parent_path.parts
            if "runs" in parent_parts:
                raise ValueError(
                    f"Cannot inherit from a run config. "
                    f"'{config_path.name}' tried to inherit from '{parent_path}'. "
                    f"Run configs can only inherit from base configs."
                )

            if config_type == "runs" and "base" not in parent_parts:
                raise ValueError(
                    f"Run configs must inherit from base configs. "
                    f"'{config_path.name}' tried to inherit from '{parent_path}'."
                )

            # load parent config recursively
            parent_config = load_config(
                str(parent_path), _visited.copy(), _is_run_config
            )
            merged_base = deep_update(merged_base, parent_config)

        # merge current config on top of parents
        config = deep_update(merged_base, config)

    return config
