import sys, os, json, re, time, datetime, logging, traceback, ast
from pathlib import Path
from typing import Any

project_name = "unithg_bmi"
proj_folder = Path("unithg_bmi")
this_folder = Path(".")
requirements = []
author_name = "Chad Perry"
author_github_link = "https://github.com/chp2001"
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
        "description": "A Python-based Ngen-BMI model for unit hydrograph.",
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

# build README.md
from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional, TypeVar
from abc import ABC, abstractmethod
class Markdown(ABC):
    format_type:str = "N/A"
    data:Any
    addable:bool = False
    def __init__(self, data:Optional[Any]=None):
        self.data = data
    @abstractmethod
    def build(self, indent:int=0)->str:
        pass
    def __str__(self)->str:
        return self.build()
    def __repr__(self)->str:
        return f"{self.format_type}({self.data})"
    def multiline_indent(self, text:str, indent:int)->str:
        return "\n".join([f"{' ' * indent}{line}" for line in text.split("\n")])
class Text(Markdown):
    format_type:str = "Text"
    data:List[str]
    addable:bool = True
    def __init__(self, data:List[str]=None):
        self.data = []
        if isinstance(data, str):
            self.data.append(data)
        elif isinstance(data, list):
            self.data.extend(data)
    def build(self, indent:int=0)->str:
        data = [item if isinstance(item, str) else item.build(indent) for item in self.data]
        return "".join(data)
    def __add__(self, other:Union[str, Markdown]):
        self.data.append(other)
        return self
    def __iadd__(self, other:Union[str, Markdown]):
        self.data.append(other)
        return self
    def __radd__(self, other:Union[str, Markdown]):
        if isinstance(other, str):
            other = Text([other])
        return other + self
    def __len__(self)->int:
        return len(self.data)
class Propagate(Markdown):
    """Markdown element that has the infrastructure to concatenate with strings."""
    def __add__(self, other:Union[str, Markdown]):
        if isinstance(other, str):
            return Text([self, other])
        elif isinstance(other, Text):
            other.data.insert(0, self)
            return other
        return self
    def __iadd__(self, other:Union[str, Markdown]):
        return self + other
    def __radd__(self, other:Union[str, Markdown]):
        if isinstance(other, str):
            other = Text([other])
        return other + self
class Header(Propagate):
    """Markdown header. Indent determines the number of '#' characters."""
    format_type:str = "Header"
    data:str
    indent: int
    def __init__(self, data:str, indent:int=2):
        self.data = data
        self.indent = indent
    def build(self, _:int=None)->str:
        header_str = "#" * min(self.indent, 4) # Max header level is 4
        return f"{header_str} {self.data}\n"
class ListItem(Markdown):
    """Unordered listitem."""
    format_type:str = "ListItem"
    data:Markdown
    marker:str = "*"
    indent:int
    def __init__(self, data:Markdown, marker:str="*", indent:Optional[int]=None):
        self.data = data
        self.marker = marker
        self.indent = indent if indent is not None else 0
    def build(self, indent:Optional[int]=None)->str:
        if indent is None:
            indent = self.indent
        return f"{'    ' * indent}{self.marker} {self.data}\n"
class OrderedListItem(ListItem):
    """Ordered listitem."""
    format_type:str = "OrderedListItem"
    marker:str = "1."
    indent:int
    data:Markdown
    def __init__(self, data:Markdown, indent:int=0):
        super().__init__(data, indent=indent)
        self.marker = "1."
    def order(self, order:Union[str,int]):
        if isinstance(order, int):
            self.marker = f"{order}."
        else:
            self.marker = order
class UnOrderedList(Markdown):
    """Unordered list."""
    format_type:str = "UnOrderedList"
    data:List[ListItem]
    addable:bool = True
    def __init__(self, data:Optional[Union[str, ListItem, List[ListItem]]]=None):
        self.data = []
        if isinstance(data, str):
            self.data.append(ListItem(data))
        elif isinstance(data, ListItem):
            self.data.append(data)
        elif isinstance(data, list):
            self.data.extend(data)
    def build(self, indent:Optional[int]=None)->str:
        result = "".join([item.build(indent) for item in self.data])
        result = result.rstrip("\n") + "\n"
        result = result.lstrip("\n") + "\n"
        return result
    def __add__(self, other:Union[str, ListItem])->'UnOrderedList':
        if not isinstance(other, ListItem):
            other = ListItem(other)
        self.data.append(other)
        return self
    def __iadd__(self, other:Union[str, ListItem])->'UnOrderedList':
        return self + other
    def __len__(self)->int:
        return len(self.data)
class OrderedList(UnOrderedList):
    """Ordered list."""
    format_type:str = "OrderedList"
    data:List[OrderedListItem]
    addable:bool = True
    def __init__(self, data:Optional[Union[str, OrderedListItem, List[OrderedListItem]]]=None):
        self.data = []
        if isinstance(data, str):
            self.data.append(OrderedListItem(data))
        elif isinstance(data, OrderedListItem):
            self.data.append(data)
        elif isinstance(data, list):
            self.data.extend(data)
    def build(self, _:Optional[int]=None)->str:
        result = ""
        marker_levels = [0] * (max([item.indent for item in self.data] + [0]) + 1)
        for listitem in self.data:
            indent = listitem.indent
            marker_levels[indent] += 1
            marker_levels[indent + 1:] = [0] * (5 - indent)
            # marker = ".".join([str(marker_levels[j]) for j in range(indent + 1)])
            marker = f"{marker_levels[indent]}"
            listitem.order(marker + ".")
            result += listitem.build()
        return result
    def __add__(self, other:Union[str, OrderedListItem])->'OrderedList':
        if not isinstance(other, OrderedListItem):
            other = OrderedListItem(other)
        self.data.append(other)
        return self
    def __iadd__(self, other:Union[str, OrderedListItem])->'OrderedList':
        return self + other
class Link(Propagate):
    """Markdown link."""
    format_type:str = "Link"
    data:Tuple[str, str]
    def build(self, indent:int=0)->str:
        return f"[{self.data[0]}]({self.data[1]})"
class SectionLink(Link):
    """Markdown section link."""
    format_type:str = "SectionLink"
    data:Tuple[str, str]
    def __init__(self, data:str):
        self.data = (data, "#" + data.lower().replace(" ", "-"))
    def build(self, indent:int=0)->str:
        return f"[{self.data[0]}]({self.data[1]})"
class Image(Markdown):
    """Markdown image."""
    format_type:str = "Image"
    data:Tuple[str, str]
    def build(self, indent:int=0)->str:
        return f"![{self.data[0]}]({self.data[1]})"
class Bold(Propagate):
    """Markdown bold text."""
    format_type:str = "Bold"
    data:str
    def build(self, indent:int=0)->str:
        return f"**{self.data}**"
class Italic(Propagate):
    """Markdown italic text."""
    format_type:str = "Italic"
    data:str
    def build(self, indent:int=0)->str:
        return f"*{self.data}*"
class Code(Propagate):
    """Markdown code segment."""
    format_type:str = "Code Segment"
    data:str
    def build(self, indent:int=0)->str:
        return f"`{self.data}`"
class Underline(Propagate):
    """Markdown underline text."""
    format_type:str = "Underline"
    data:str
    def build(self, indent:int=0)->str:
        return f"<u>{self.data}</u>"
class Paragraph(Markdown):
    """Markdown paragraph."""
    format_type:str = "Paragraph"
    data:List[Text]
    addable:bool = True
    def __init__(self, data:Optional[Union[str, Text, List[Text]]]=None):
        self.data = []
        if isinstance(data, str):
            self.data.append(Text([data]))
        elif isinstance(data, Text):
            self.data.append(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    self.data.append(Text([item]))
                else:
                    self.data.append(item)
    def build(self, indent:int=0)->str:
        result = "".join([item.build(indent) for item in self.data])
        result = result.rstrip("\n") + "\n\n"
        return result
    def __add__(self, other:Union[str, Markdown])->'Paragraph':
        if isinstance(other, Paragraph):
            self.data.extend(other.data)
            return self
        elif isinstance(other, str):
            other = Text([other])
        self.data.append(other)
        return self
    def __iadd__(self, other:Union[str, Markdown])->'Paragraph':
        return self + other
class CodeBlock(Paragraph):
    """Markdown code block."""
    format_type:str = "Code Block"
    data:List[Text]
    language:str
    def __init__(self, data:Any=None, language:str="python"):
        super().__init__(data=data)
        self.language = language
    def build(self, indent:int=0)->str:
        result = f"```{self.language}\n"
        middle = "\n".join([item.build(indent) for item in self.data])
        if indent > 0:
            middle = self.multiline_indent(middle, indent)
        return result + middle + "\n```\n"
class Section(Markdown):
    """Markdown section."""
    format_type:str = "Section"
    data:List[Markdown]
    header:Header
    name:str
    indent:int
    addable:bool = True
    def __init__(self, name:str, data:Optional[List[Markdown]]=None, indent:int=2):
        self.header = Header(name, indent)
        self.name = name
        self.data = data if data is not None else []
        self.indent = indent
    def build(self, indent:int=0)->str:
        result = self.header.build(indent) + "\n" + "".join([item.build(indent) for item in self.data])
        result = result.rstrip("\n") + "\n\n"
        return result
    def __add__(self, other:Union[str, Markdown]):
        if isinstance(other, str):
            other = Paragraph(other)
        self.data.append(other)
        return self
    def __iadd__(self, other:Union[str, Markdown]):
        return self + other
    def __len__(self)->int:
        return len(self.data)
    def get_link(self)->SectionLink:
        return SectionLink(self.name)
    
class TableOfContents(Section):
    """Markdown table of contents."""
    format_type:str = "TableOfContents"
    data:List[Markdown]
    def __init__(self, data:List[Tuple[int, SectionLink]]=None):
        self.data = [OrderedList()]
        self.header = Header("Table of Contents", 2)
        self.name = "Table of Contents"
        self.indent = 2
    def add(self, link:SectionLink, indent:int)->None:
        if link.data[0] == "Table of Contents":
            return
        indent = max(indent, 2) - 2
        self.data[0] += OrderedListItem(link, indent)
    def link(self, section:Section)->None:
        self.add(section.get_link(), section.indent)
        
class MarkdownFile:
    titleHeader:Optional[Header]
    sections:List[str]
    content:List[Section]
    table_of_contents:TableOfContents
    def __init__(self, sections:Optional[List[str]] = None):
        self.sections = sections if sections is not None else []
        self.content = []
        self.table_of_contents = TableOfContents([])
    def add_section(self, section:Section):
        self.content.append(section)
        self.table_of_contents.link(section)
        if section.name not in self.sections:
            self.sections.append(section.name)
    def add(self, section: Union[Section, str]):
        if isinstance(section, str):
            if section == "Table of Contents":
                return self.add_section(self.table_of_contents)
            raise ValueError(f"Invalid section: {section}")
        elif isinstance(section, Header):
            # section = Section(name=section.data, indent=section.indent)
            self.titleHeader = section
            return
        self.add_section(section)
    def build(self)->str:
        result = ""
        result += self.titleHeader.build() + "\n"
        for section in self.content:
            # print(f"Building section: {section.name}, {section.data}")
            section_str = section.build()
            result += section_str
        result = result.rstrip("\n") + "\n"
        return result
    def to_file(self, path:Path):
        path.write_text(self.build())
        
    
    
readme_path = this_folder/"README.md"
readme_sections = [
    "Project Title",
    "Description",
    "Table of Contents",
    "Dependencies",
    "Installation",
    "Usage",
    "Testing",
    "Known Issues",
    "Getting Help",
    "Getting Involved",
    "Open Source Licensing Info",
    "Credits and References"
]

def author_link()->Link:
    return Link((author_name, author_github_link))

def build_project_title()->Header:
    return Header(project_name, 1)

def build_description(indent:int = 2)->Section:
    desc = Section("Description")
    desc += "This project is a Python-based Nextgen-BMI model that implements unit hydrograph functionality."
    return desc

def build_dependencies(indent:int = 2)->Section:
    deps = Section("Dependencies", indent=indent)
    desc = Paragraph("The following dependencies are required for this project:")
    deps_list = OrderedList()
    for req in requirements:
        deps_list += req
    deps += desc
    deps += deps_list
    return deps

def build_installation(indent:int = 2)->Section:
    install = Section("Installation", indent=indent)
    desc = Paragraph("To install this project, run the following command:")
    code = CodeBlock("pip install .")
    install += desc
    install += code
    return install

def build_usage(indent:int = 2)->Section:
    usage = Section("Usage", indent=indent)
    para_text = "Using this project requires a working installation of " + Link(("Ngen", "https://github.com/NOAA-OWP/ngen")) + ","
    para_text += " or a working dev container with Ngen installed, such as through " + Link(("NGIAB-CloudInfra", "https://github.com/CIROH-UA/NGIAB-CloudInfra")) + "."
    usage += Paragraph(para_text)
    para_text = "Once you have a working installation of Ngen, you can incorporate the included " + Code("formulation.json") + " into an Ngen realization file."
    usage += Paragraph(para_text)
    para_text = "The last step is to ensure that this project is installed in the same environment as Ngen, such that it can be accessed by Ngen."
    usage += Paragraph(para_text)
    return usage

def build_testing(indent:int = 2)->Section:
    testing = Section("Testing", indent=indent)
    testing += Paragraph("This project currently does not include an automated testing suite.")
    testing += Paragraph(
        Text("However, most included functionality can be tested through the use of __name__ == '__main__' blocks, or the ") 
        + Code('test_module.py') + Text(" script."))
    return testing

def build_known_issues(indent:int = 2)->Section:
    issues = Section("Known Issues", indent=indent)
    issues += Paragraph("This project currently does not currently have any known issues.")
    return issues

def build_getting_help(indent:int = 2)->Section:
    help = Section("Getting Help", indent=indent)
    help += Paragraph(
        Text("If you have questions or need help with this project, please contact the author, ")
        + author_link()
        + Text(", or open an issue on the project's GitHub repository.")
    )
    return help

def build_getting_involved(indent:int = 2)->Section:
    involved = Section("Getting Involved", indent=indent)
    involved += Paragraph(
        Text("If you feel that something is missing or could be improved, please feel free to open an issue or a pull request.")
        + Text(" Additional details are available in the project's ")
        + Link(("CONTRIBUTING.md", "CONTRIBUTING.md"))
        + Text(" file.")
    )
    return involved

def build_open_source_licensing_info(indent:int = 2)->Section:
    licensing = Section("Open Source Licensing Info", indent=indent)
    info = OrderedList()
    # info += Link(("TERMS", "TERMS.md"))
    info += Link(("LICENSE", "LICENSE"))
    licensing += info
    return licensing

def build_credits_and_references(indent:int = 2)->Section:
    credits = Section("Credits and References", indent=indent)
    credits += Paragraph(
        Text("This project was created by ") + author_link() + Text(" on July 16, 2024.")
        )
    credits += Paragraph(
        f"This README.md file was generated by a Python script on {current_date}."
    )
    return credits

readme = MarkdownFile()
readme.add(build_project_title())
# print(readme.build())
readme.add(build_description())
# print(readme.build())
readme.add("Table of Contents")
readme.add(Section("Setup", indent=2))
readme.add(build_dependencies(3))
readme.add(build_installation(3))
readme.add(build_usage())
readme.add(build_testing())
readme.add(Section("Participation", indent=2))
readme.add(build_known_issues(3))
readme.add(build_getting_help(3))
readme.add(build_getting_involved(3))
readme.add(build_open_source_licensing_info())
readme.add(build_credits_and_references())

output_path = this_folder/"README.md"

output_path.write_text(readme.build())