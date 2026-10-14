"""Importable helper: seed ~/.libero/config.yaml so LIBERO never calls input()."""
import os, yaml

def seed_libero_config(libero_pkg="/workspace/LIBERO/libero/libero"):
    cfg_dir = os.path.expanduser("~/.libero")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump({
            "benchmark_root": libero_pkg,
            "bddl_files": os.path.join(libero_pkg, "bddl_files"),
            "init_states": os.path.join(libero_pkg, "init_files"),
            "datasets": os.path.join(libero_pkg, "../datasets"),
            "assets": os.path.join(libero_pkg, "assets"),
        }, f)
    return cfg_path

if __name__ == "__main__":
    print("seeded:", seed_libero_config())
