import sys, os, json, re, time, datetime, logging, traceback, ast
from pathlib import Path
from typing import Any

project_name = "pyflo_bmi"
proj_folder = Path("pyflo_bmi")
this_folder = Path(".")
requirements = []
author_name = "Chad Perry"
# Month Day, Year
current_date = datetime.datetime.now().strftime("%B %d, %Y")

def harvest_requirements(src_dir:Path = proj_folder, requirements:list[str] = []):
    for name in src_dir.iterdir():
        if name.is_dir():
            harvest_requirements(name, requirements)
        elif name.suffix == ".py":
            tree = ast.parse(name.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == ".":
                            continue
                        requirements.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module == None:
                        continue
                    requirements.append(node.module)
    return requirements

def project_filenames(src_dir:Path = proj_folder, filenames:list[str] = []):
    for name in src_dir.iterdir():
        if name.is_dir():
            project_filenames(name, filenames)
        elif name.suffix == ".py":
            filenames.append(name.name)
    return filenames

project_files = project_filenames()
_requirements = harvest_requirements()
for req in _requirements:
    _req = req
    if req in sys.builtin_module_names or req in sys.stdlib_module_names:
        continue
    if req == project_name:
        continue
    if req.startswith("."):
        continue
    if "." in req:
        req = req.split(".")[0]
    if (proj_folder/req).exists() or (proj_folder/(req + ".py")).exists():
        continue
    if req + ".py" in project_files:
        continue
    print(f"Adding requirement: {_req}")
    requirements.append(req)
requirements = list(set(requirements))
requirements.sort()
print(requirements)

def build_requirements_txt(requirements:list[str], path:Path = this_folder/"requirements.txt"):
    path.write_text("\n".join(requirements))

build_requirements_txt(requirements)

def build_pyproject_toml(project_name:str, requirements:list[str], path:Path = this_folder/"pyproject.toml"):

    toml = {}
    toml["build-system"] = {
        "requires": ["setuptools", "wheel"],
        "build-backend": "setuptools.build_meta"
    }
    toml["project"] = {
        "name": project_name,
        "version": "0.1.0",
        "description": "A Python-based Ngen-BMI model for the PyFlo module.",
        "authors": [{"name": author_name}],
        "maintainers": [{"name": author_name}],
        "license": {"file": "LICENSE"},
        "requires-python": ">=3.9",
    }
    dependencies = requirements
    toml["project"]["dependencies"] = dependencies
    # toml["project"]["dynamic"] = ["readme"]

    toml["tool.setuptools.packages.find"] = {"where": ["."]}
    
    def toml_str(toml:dict):
        def toml_val_len(val:Any):
            if isinstance(val, list):
                return sum([toml_val_len(v) for v in val])
            elif isinstance(val, dict):
                return sum([1 + toml_val_len(v) for k, v in val.items()])
            return 1
        def toml_val_str(val:Any):
            if isinstance(val, str):
                return f'"{val}"'
            elif isinstance(val, list):
                len_val = toml_val_len(val)
                if len_val == 0:
                    return "[]"
                if len_val <= 2:
                    # Inline list
                    return f"[{', '.join([toml_val_str(v) for v in val])}]"
                # Multiline list
                result = "[\n"
                indent = "\t"
                for v in val:
                    result += f"{indent}{toml_val_str(v)},\n"
                result += "]"
                return result
            elif isinstance(val, dict):
                return f"{{{', '.join([f'{k} = {toml_val_str(v)}' for k, v in val.items()])}}}"
            return str(val)
        cats = []
        for cat, val in toml.items():
            cat_str = f"[{cat}]\n"
            # val is always a dict
            val:dict[str, Any]
            for key, val in val.items():
                cat_str += f"{key} = {toml_val_str(val)}\n"
            cats.append(cat_str)
        return "\n".join(cats)
    
    path.write_text(toml_str(toml))

build_pyproject_toml(project_name, requirements)

def build_init_py(path:Path = proj_folder/"__init__.py"):
    imports = []
    for name in proj_folder.iterdir():
        if name.is_dir():
            prefix = name.name
            for subname in name.iterdir():
                if subname.suffix == ".py":
                    imports.append(f"from .{prefix}.{subname.stem} import *")
        elif name.suffix == ".py":
            if name.stem == "__init__":
                continue
            imports.append(f"from .{name.stem} import *")
    path.write_text("\n".join(imports))

build_init_py()

