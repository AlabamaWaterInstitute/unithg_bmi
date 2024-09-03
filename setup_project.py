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
from typing import List, Tuple, Dict, Set, Any, Union, Callable, Literal, Optional, TypeVar, _SpecialForm
@_SpecialForm
def Listable(self, param:type):
    """
    Listable[arg] -> Union[List[arg], arg]
    Allows annotating a type as either a list of a type or the type itself.
    """
    arg = param
    if not isinstance(arg, type) and not "typing" in str(arg):
        raise TypeError(f"Listable[...] expects a type, not {arg}(type={type(arg)})")
    return Union[List[arg], arg]
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
    @abstractmethod
    def __len__(self)->int: pass
    @abstractmethod
    def __height__(self, maxwidth:Optional[int] = None)->int: pass
    def __width__(self)->int:
        return len(self)
    def size(self)->int:
        """Total expected text amount."""
        if isinstance(self.data, Markdown):
            return self.data.size()
        elif isinstance(self.data, list):
            size = 0
            for item in self.data:
                if isinstance(item, Markdown):
                    size += item.size()
                else:
                    size += len(str(item))
            return size
        return len(str(self.data))
def width(markdown:Markdown)->int:
    return markdown.__width__() if hasattr(markdown, "__width__") else len(markdown)
class SingleItem(Markdown):
    """
    Abstract base class for Markdown elements that contain a single item/line.
    
    Purpose is to provide a common height and width method for relevant Markdown elements.
    """
    words:List[str]
    def __width__(self)->int:
        _width = 0
        if hasattr(self.data, "__iter__"):
            for item in self.data:
                if isinstance(item, Markdown):
                    _width += width(item)
                else:
                    _width += len(str(item))
        else:
            _width = width(self.data)
        return _width
    def __height__(self, maxwidth:Optional[int] = None)->int:
        if maxwidth is None:
            return 1
        height = 0
        xsize = 0
        for item in self.data:
            if isinstance(item, Markdown):
                _width = width(item)
            else:
                _width = len(str(item))
            if xsize + _width > maxwidth:
                height += 1
                xsize = 0
            xsize += _width
        return height
    def __len__(self)->int:
        return self.__width__()
class MultiItem(Markdown):
    """
    Abstract base class for Markdown elements that contain multiple items/lines.
    """
    def __width__(self)->int:
        return max([width(item) for item in self.data])
    def __height__(self, maxwidth:Optional[int] = None)->int:
        if maxwidth is None:
            return len(self.data)
        return sum([item.__height__(maxwidth) for item in self.data])
    def __len__(self)->int:
        return len(self.data)
class Text(SingleItem):
    format_type:str = "Text"
    data:List[Union[str, Markdown]]
    words:List[str]
    addable:bool = True
    def __init__(self, data:List[Union[str, Markdown]]):
        self.words = []
        self.data = []
        self.add(data)
    def add(self, text:Union[str, Markdown, List[Union[str, Markdown]]]):
        if isinstance(text, list):
            for item in text:
                self.add(item)
            return
        self.data.append(text)
        if isinstance(text, str):
            self.words.extend(text.split())
        elif isinstance(text, SingleItem):
            self.words.extend(text.words)
        else:
            raise NotImplementedError(f"Unsupported text type: {type(text)}")
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
                
class Propagate(SingleItem):
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
    
class ListItem(SingleItem):
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
    def __width__(self)->int:
        return max([width(item) for item in self.data])
    def __height__(self, maxwidth:Optional[int]=None)->int:
        if maxwidth is None:
            return len(self.data)
        return sum([item.__height__(maxwidth) for item in self.data])
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
    def __len__(self)->int:
        return len(self.data)
    def __width__(self)->int:
        return max([width(item) for item in self.data])
    def __height__(self, maxwidth:Optional[int]=None)->int:
        if maxwidth is None:
            return len(self.data)
        return sum([item.__height__(maxwidth) for item in self.data])
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
class Paragraph(MultiItem):
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
class Section(MultiItem):
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
    def __init__(self, data:List[Tuple[SectionLink, int]]=None):
        self.data = [OrderedList()]
        self.header = Header("Table of Contents", 2)
        self.name = "Table of Contents"
        self.indent = 2
        if data is not None:
            for item in data:
                self.add(*item)
    def add(self, link:SectionLink, indent:int)->None:
        if link.data[0] == "Table of Contents":
            return
        indent = max(indent, 2) - 2
        self.data[0] += OrderedListItem(link, indent)
    def link(self, section:Section)->None:
        self.add(section.get_link(), section.indent)

class MarkdownTable: pass # Table container / parent
class MarkdownTableSeries: pass # Base class for table rows and columns
class MarkdownTableRow: pass # Table row (buildable)
class MarkdownTableColumn: pass # Table column (data only)
class MarkdownTableCell: pass # Table cell
class MarkdownTable: # partial definition
    columns:List[MarkdownTableColumn]
    rows:List[MarkdownTableRow]
    data:List[MarkdownTableCell]
    def get_columns(self)->List[MarkdownTableColumn]: pass
    def get_rows(self)->List[MarkdownTableRow]: pass
    def column(self, index:int)->MarkdownTableColumn: pass
    def row(self, index:int)->MarkdownTableRow: pass
    def cell(self, row:int, col:int)->MarkdownTableCell: pass
    def calculate_max_widths(self, total_max_width:int)->List[int]: pass
    pass
SeriesIndexType = Union[int, str, slice]
class MarkdownTableSeries(ABC):
    data:List[MarkdownTableCell]
    parent:MarkdownTable
    index:int
    axis:Literal["row", "column"]
    @abstractmethod
    def width(self)->int: pass
    def add(self, cell:MarkdownTableCell)->None:
        self.data.append(cell)
    def __index__(self, index:SeriesIndexType)->Union[int, slice]:
        if isinstance(index, str):
            if self.axis == "row":
                return self.parent.columns.index(index)
            raise ValueError(f"Cannot index column by column name: {index}")
        elif isinstance(index, slice):
            args = [index.start, index.stop, index.step]
            args = [self.__index__(arg) if arg is not None else arg for arg in args]
            return slice(*args)
        else:
            return index
    def __getitem__(self, index:SeriesIndexType)->Listable[MarkdownTableCell]:
        index = self.__index__(index)
        return self.data[index]
    def __setitem__(self, index:SeriesIndexType, data:Listable[MarkdownTableCell])->None:
        index = self.__index__(index)
        if isinstance(index, slice):
            if isinstance(data, list):
                self.data[index] = data
            else:
                self.data[index] = [data] * len(index)
        else:
            if isinstance(data, list):
                raise ValueError("Cannot set single cell to multiple cells.")
            self.data[index] = data
    def __len__(self)->int:
        return len(self.data)
        
    
class MarkdownTableCell:
    """
    Markdown table cell.
    Can calculate its own width and height based on the data it contains.
    """
    data:Markdown
    rownum:Optional[int]
    colnum:Optional[int]
    def __init__(self, data:Any, rownum:Optional[int]=None, colnum:Optional[int]=None):
        if isinstance(data, str):
            self.data = Text([data])
        elif isinstance(data, Markdown):
            self.data = data
        else:
            self.data = Text([str(data)])
        self.rownum = rownum
        self.colnum = colnum
    def width(self)->int:
        if isinstance(self.data, str):
            return len(self.data)
        return width(self.data)
    def height(self, maxwidth:Optional[int]=None)->int:
        if isinstance(self.data, str):
            return 1
        return self.data.__height__(maxwidth)
    def __str__(self)->str:
        return str(self.data)
    def __repr__(self)->str:
        return f"MarkdownTableCell({self.data})"
    def set_rownum(self, rownum:int)->None:
        self.rownum = rownum
    def set_colnum(self, colnum:int)->None:
        self.colnum = colnum
    def __len__(self)->int:
        return self.width()
    def build(self, desiredwidth:int, desiredheight:int)->str:
        result = ""
        if isinstance(self.data, str):
            result = self.data
        else:
            result = self.data.build()
        lines = result.split("\n")
        if len(lines) < desiredheight:
            lines += [""] * (desiredheight - len(lines))
        result = "\n".join([line.ljust(desiredwidth) for line in lines])
        return result
    def data_len(self)->int:
        if isinstance(self.data, str):
            return len(self.data)
        if isinstance(self.data, Markdown):
            return self.data.size()
        return len(str(self.data))
    
class MarkdownTableRow(MarkdownTableSeries):
    """
    Markdown table row.
    """
    data:List[MarkdownTableCell]
    def __init__(self, data:Optional[List[MarkdownTableCell]] = None, parent:MarkdownTable = None, index:Optional[int] = None):
        if parent is None:
            raise ValueError("Parent table must be provided.")
        if index is None:
            index = len(parent.rows)
        if data is None:
            data = []
        self.data = data
        self.parent = parent
        self.index = index
        self.axis = "row"
    def width(self, maxwidth:Optional[int]=None)->int:
        return sum([min(cell.width(), maxwidth) for cell in self.data])
    def height(self, maxwidth:Optional[int]=None)->int:
        return max([cell.height(maxwidth) for cell in self.data])
    def __len__(self)->int:
        return self.width()
    def build(self, desired_widths:List[int], divider:str = "|")->str:
        result_lines = []
        for i, cell in enumerate(self.data):
            desired_width = desired_widths[i]
            cell_str = cell.build(desired_width, self.height())
            cell_lines = cell_str.split("\n")
            for j, line in enumerate(cell_lines):
                if j >= len(result_lines):
                    result_lines.append([])
                append = f"{divider} {line}"
                if i > 0:
                    append = f" {append}"
                result_lines[j].append(append)
        for i in range(len(result_lines)):
            result_lines[i] = "".join(result_lines[i]) + " |\n"
        return "".join(result_lines)
    
class MarkdownTableColumn(MarkdownTableSeries):
    """
    Markdown table column.
    """
    data:List[MarkdownTableCell]
    def __init__(self, data:Optional[List[MarkdownTableCell]] = None, parent:MarkdownTable = None, index:Optional[int] = None):
        if parent is None:
            raise ValueError("Parent table must be provided.")
        if index is None:
            index = len(parent.columns)
        if data is None:
            data = []
        self.data = data
        self.parent = parent
        self.index = index
        self.axis = "column"
    def width(self)->int:
        return max([cell.width() for cell in self.data])
    def height(self, maxwidth:Optional[int]=None)->int:
        return sum([cell.height(maxwidth) for cell in self.data])
    def __len__(self)->int:
        return self.width()
    def build(self, desired_width:int, desired_height:int, divider:str = "|")->str:
        """Column build won't be used outside of debugging. The row build method handles everything."""
        result = ""
        for cell in self.data:
            result += cell.build(desired_width, desired_height) + "\n"
        return result
    def weight(self)->int:
        weight = 0
        for cell in self.data:
            weight += cell.data_len()
        return weight
    
class MarkdownHeaderCell(MarkdownTableCell):
    """
    Markdown table header cell.
    Identical to the standard cell, but a bottom border is added.
    """
    def build(self, desiredwidth:int, desiredheight:int)->str:
        result = super().build(desiredwidth, desiredheight)
        result += "\n"
        result += "-" * desiredwidth
        return result
    
class MarkdownHeaderRow(MarkdownTableRow):
    """
    Markdown table header row.
    """
    data:List[MarkdownHeaderCell]
    def __init__(self, data:List[MarkdownHeaderCell], parent:MarkdownTable):
        super().__init__(data, parent, 0)

TableIndexerType = Union[int, tuple["TableIndexerType", "TableIndexerType"], slice, str]
class MarkdownTable(Markdown):
    """
    Markdown table.
    """
    column_names:List[str]
    header_row:MarkdownHeaderRow
    columns:List[MarkdownTableColumn]
    rows:List[MarkdownTableRow]
    data:List[MarkdownTableCell]
    maxwidth:int
    def __init__(self, maxwidth:int = 40, column_names:List[str] = [], data:Optional[List[List[Union[str, Markdown]]]]=None):
        self.column_names = column_names
        self.maxwidth = maxwidth
        self.columns = []
        self.rows = []
        self.header_row = MarkdownHeaderRow([MarkdownHeaderCell(name) for name in column_names], self)
        for i, name in enumerate(column_names):
            self.columns.append(MarkdownTableColumn(parent=self))
            self.columns[i].add(self.header_row.data[i])
        self.rows.append(self.header_row)
        self.data = []
        if data is not None:
            for i, row in enumerate(data):
                self.add_row(row, i + 1)
    def get_columns(self)->List[MarkdownTableColumn]:
        return self.columns
    def get_rows(self)->List[MarkdownTableRow]:
        return self.rows
    def column(self, index:Union[str, int])->MarkdownTableColumn:
        if isinstance(index, str):
            return self.columns[self.column_names.index(index)]
        return self.columns[index]
    def row(self, index:int)->MarkdownTableRow:
        return self.rows[index]
    def __index__(self, index:TableIndexerType)->Union[int, slice, tuple]:
        if isinstance(index, str):
            return self.column_names.index(index)
        elif isinstance(index, tuple):
            col = self.__index__(index[0])
            row = self.columns[0].__index__(index[1])
            return (col, row)
        elif isinstance(index, slice):
            return self.header_row.__index__(index)
        return index
    def __getitem__(self, index:TableIndexerType)->Listable[Union[MarkdownTableCell, MarkdownTableRow, MarkdownTableColumn]]:
        index = self.__index__(index)
        # [:, :] -> List[MarkdownTableCell]
        # [:] -> List[MarkdownTableColumn]
        # [:, 0] -> MarkdownTableRow
        # [0, :] -> MarkdownTableColumn
        # [0, 1:] -> List[MarkdownTableCell]
        # [1:, 0] -> List[MarkdownTableRow]
        # [1:, 1:] -> List[List[MarkdownTableCell]]
        if isinstance(index, slice):
            # [:] -> List[MarkdownTableColumn]
            return self.columns[index]
        elif isinstance(index, tuple):
            # [0|:, 0|:] -> ?
            if isinstance(index[0], int):
                # [0, 0|:] -> Listable[MarkdownTableCell]
                col = self.columns[index[0]]
                return col[index[1]]
            # [:, 0|:] -> Listable[Listable[MarkdownTableCell]]
            cols = self.columns[index[0]]
            result = []
            for col in cols:
                result.append(col[index[1]])
            return result
        elif isinstance(index, int):
            # [0] -> MarkdownTableColumn
            return self.columns[index]
        else:
            raise IndexError(f"Unknown index type: {index}(type={type(index)})")
    def __setitem__(self, index:TableIndexerType, data:Listable[Union[MarkdownTableCell, MarkdownTableRow, MarkdownTableColumn]])->None:
        index = self.__index__(index)
        if isinstance(index, slice):
            if isinstance(data, list):
                self.columns[index] = data
            else:
                self.columns[index] = [data] * len(index)
        elif isinstance(index, tuple):
            if isinstance(data, list):
                for i, col in enumerate(self.columns[index[0]]):
                    col[index[1]] = data[i]
            else:
                self.columns[index[0]][index[1]] = data
        elif isinstance(index, int):
            self.columns[index] = data
        else:
            raise IndexError(f"Unknown index type: {index}(type={type(index)})")
    def calculate_max_widths(self, total_max_width:int)->List[int]:
        expected_widths = [max([len(name)] + [self.columns[i].width()]) for i, name in enumerate(self.column_names)]
        expected_max_width = sum(expected_widths)
        if expected_max_width <= total_max_width:
            return expected_widths
        weights = [col.weight() for col in self.columns]
        total_weight = sum(weights)
        ratio = total_max_width / expected_max_width
        expected_avg_width = total_max_width // len(self.column_names)
        expected_widths = [int(expected_avg_width * weight / total_weight) for weight in weights]
        return expected_widths
    def build(self, indent:int=0)->str:
        maxwidth = max(self.maxwidth, 20 * len(self.column_names))
        widths = self.calculate_max_widths(maxwidth)
        # result = self.header_row.build(widths)
        result = ""
        for row in self.rows:
            result += row.build(widths)
        return result
    def __len__(self)->int:
        return self.header_row.width()
    def __new_row(self)->MarkdownTableRow:
        result = MarkdownTableRow(parent=self)
        self.rows.append(result)
        return result
    def __new_column(self, name:str)->MarkdownTableColumn:
        result = MarkdownTableColumn(parent=self)
        self.columns.append(result)
        self.column_names.append(name)
        return result
    def add_cell(self, data:Union[str, Markdown], row:int, col:int)->None:
        maxrow = len(self.rows)
        maxcol = len(self.columns)
        targetrow = None
        if row < maxrow:
            targetrow = self.rows[row]
        elif row == maxrow:
            targetrow = self.__new_row()
        else:
            raise IndexError(f"Row index out of range: {row} (max={maxrow}).\nCell islands are not supported.")
        if col >= maxcol:
            raise IndexError(f"No corresponding column for index: {col} (max={maxcol}).")
        targetcol = self.columns[col]
        cell = MarkdownTableCell(data, row, col)
        targetrow.add(cell)
        targetcol.add(cell)
        self.data.append(cell)
    def add_row(self, data:List[Union[str, Markdown]])->None:
        row_index = len(self.rows)
        for i, item in enumerate(data):
            self.add_cell(item, row_index, i)
    def add_column(self, name:str, data:List[Union[str, Markdown]])->None:
        col_index = len(self.columns)
        col = self.__new_column(name)
        for i, item in enumerate(data):
            self.add_cell(item, i, col_index)
    def add(self, data:List[List[Union[str, Markdown]]])->None:
        for row in data:
            self.add_row(row)
            
    def __width__(self) -> int:
        return min(self.maxwidth, sum(col.width() for col in self.columns))
    def __height__(self, maxwidth:Optional[int] = None) -> int:
        return sum(row.__height__(maxwidth) for row in self.rows)
    
        
class MarkdownFile:
    titleHeader:Optional[Header]
    sections:List[str]
    content:List[Section]
    table_of_contents:TableOfContents
    titleHeader:Optional[Header]
    def __init__(self, sections:Optional[List[str]] = None):
        self.sections = sections if sections is not None else []
        self.content = []
        self.table_of_contents = TableOfContents([])
        self.titleHeader = None
    def add_section(self, section:Section):
        self.content.append(section)
        self.table_of_contents.link(section)
        if section.name not in self.sections:
            self.sections.append(section.name)
    def add(self, section: Union[Section, str, List[Section]]):
        if isinstance(section, str):
            if section == "Table of Contents":
                return self.add_section(self.table_of_contents)
            raise ValueError(f"Invalid section: {section}")
        elif isinstance(section, Header):
            if len(self.content) > 0:
                section = Section(name=section.data, indent=section.indent)
            else:
                self.titleHeader = section
                return
        elif isinstance(section, list):
            for sec in section:
                self.add(sec)
            return
        self.add_section(section)
    def build(self)->str:
        result = ""
        if self.titleHeader is not None:
            result += self.titleHeader.build() + "\n"
        for section in self.content:
            # print(f"Building section: {section.name}, {section.data}")
            section_str = section.build()
            result += section_str
        result = result.rstrip("\n") + "\n"
        return result
    def to_file(self, path:Path):
        path.write_text(self.build())
        
install_md_path = this_folder/"INSTALL.md"
        
    
    
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

## Links
ngen_address = "https://github.com/NOAA-OWP/ngen"
ngiab_cloudinfra_address = "https://github.com/CIROH-UA/NGIAB-CloudInfra"
alabama_water_institute_address = "https://github.com/AlabamaWaterInstitute/"
bmi_reference_address = "https://github.com/NOAA-OWP/ngen/blob/master/doc/BMIconventions.md"

def author_link()->Link:
    return Link((author_name, author_github_link))

def ngen_link()->Link:
    return Link(("Ngen", ngen_address))

def nextgen_link()->Link:
    return Link(("Nextgen", ngen_address))

def ngiab_cloudinfra_link()->Link:
    return Link(("NGIAB-CloudInfra", ngiab_cloudinfra_address))

def alabama_water_institute_link()->Link:
    return Link(("Alabama Water Institute", alabama_water_institute_address))

def bmi_reference_link()->Link:
    return Link(("BMI", bmi_reference_address))

## README.md sections

def build_project_title()->Header:
    return Header(project_name, 1)

def build_description(indent:int = 2)->Section:
    desc = Section("Description")
    # para = Paragraph("This project is a Python-based Nextgen-BMI model that implements unit hydrograph functionality.")
    para = "This project is a Python-based"
    para += " " + bmi_reference_link() + " model that implements unit hydrograph functionality"
    para += " for use with " + nextgen_link() + "."
    desc += Paragraph(para)
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
    desc = Paragraph("To install this project, run the following command in the project directory:")
    code = CodeBlock("pip install .")
    install += desc
    install += code
    return install

def build_usage(indent:int = 2)->Section:
    usage = Section("Usage", indent=indent)
    para_text = "Using this project requires a working installation of " + ngen_link() + ","
    para_text += " or a working dev container with Ngen installed, such as through " + ngiab_cloudinfra_link() + "."
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

def build_setup_section_group(headername:str = "Setup", extra_indent:int = 2)->List[Section]:
    return [
        Header(headername, extra_indent),
        build_dependencies(indent=extra_indent + 1),
        build_installation(indent=extra_indent + 1),
        build_usage(indent=extra_indent + 1)
    ]

def build_readme_md(path:Path):
    readme_md = MarkdownFile()
    readme_md.add(build_project_title())
    readme_md.add(build_description())
    readme_md.add("Table of Contents")
    readme_md.add(Section("Setup", indent=2))
    readme_md.add(build_dependencies(3))
    readme_md.add(build_installation(3))
    readme_md.add(build_usage())
    readme_md.add(build_testing())
    readme_md.add(Section("Participation", indent=2))
    readme_md.add(build_known_issues(3))
    readme_md.add(build_getting_help(3))
    readme_md.add(build_getting_involved(3))
    readme_md.add(build_open_source_licensing_info())
    readme_md.add(build_credits_and_references())
    readme_md.to_file(path)
    
def build_install_md(path:Path):
    install_md = MarkdownFile()
    content = build_setup_section_group("Installation instructions", extra_indent=1)
    install_md.add(content)
    install_md.to_file(path)
    
build_readme_md(readme_path)
build_install_md(install_md_path)

# SECURITY.md

security_md_path = this_folder/"SECURITY.md"

def build_security_policy(indent:int = 1)->Section:
    # Security Policy

    # There MUST be no unpatched vulnerabilities of medium or higher severity that have been publicly known for more than 60 days. 

    # The vulnerability must be patched and released by the project itself (patches may be developed elsewhere). A vulnerability becomes publicly known (for this purpose) once it has a CVE with publicly released non-paywalled information (reported, for example, in the <a href="https://nvd.nist.gov/">National Vulnerability Database</a>) or when the project has been informed and the information has been released to the public (possibly by the project). A vulnerability is considered medium or higher severity if its <a href="https://www.first.org/cvss/" >Common Vulnerability Scoring System (CVSS)</a> base qualitative score is medium or higher. In CVSS versions 2.0 through 3.1, this is equivalent to a CVSS score of 4.0 or higher. Projects may use the CVSS score as published in a widely-used vulnerability database (such as the <a href="https://nvd.nist.gov">National Vulnerability Database</a>) using the most-recent version of CVSS reported in that database. Projects may instead calculate the severity themselves using the latest version of <a href="https://www.first.org/cvss/">CVSS</a> at the time of the vulnerability disclosure, if the calculation inputs are publicly revealed once the vulnerability is publicly known. <strong>Note</strong>: this means that users might be left vulnerable to all attackers worldwide for up to 60 days. This criterion is often much easier to meet than what Google recommends in <a href="https://security.googleblog.com/2010/07/rebooting-responsible-disclosure-focus.html">Rebooting responsible disclosure</a>, because Google recommends that the 60-day period start when the project is notified _even_ if the report is not public. Also note that this badge criterion, like other criteria, applies to the individual project. Some projects are part of larger umbrella organizations or larger projects, possibly in multiple layers, and many projects feed their results to other organizations and projects as part of a potentially-complex supply chain. An individual project often cannot control the rest, but an individual project can work to release a vulnerability patch in a timely way. Therefore, we focus solely on the individual project's response time.  Once a patch is available from the individual project, others can determine how to deal with the patch (e.g., they can update to the newer version or they can apply just the patch as a cherry-picked solution).

    # The public repositories MUST NOT leak any valid private credential (e.g., a working password or private key) that is intended to limit public access.
    security = Section("Security Policy", indent=indent)
    security += Paragraph(
        "There MUST be no unpatched vulnerabilities of medium or higher severity that have been publicly known for more than 60 days."
    )
    para_2 = Paragraph()
    para_2 += "The vulnerability must be patched and released by the project itself (patches may be developed elsewhere)."
    para_2 += " A vulnerability becomes publicly known (for this purpose) once it has a CVE with publicly released non-paywalled information"
    para_2 += " (reported, for example, in the " + Link(("National Vulnerability Database", "https://nvd.nist.gov/")) + ")."
    para_2 += " or when the project has been informed and the information has been released to the public (possibly by the project)."
    para_2 += " A vulnerability is considered medium or higher severity if its " + Link(("Common Vulnerability Scoring System (CVSS)", "https://www.first.org/cvss/")) + " base qualitative score is medium or higher."
    para_2 += " In CVSS versions 2.0 through 3.1, this is equivalent to a CVSS score of 4.0 or higher."
    para_2 += " Projects may use the CVSS score as published in a widely-used vulnerability database (such as the " + Link(("National Vulnerability Database", "https://nvd.nist.gov")) + ") using the most-recent version of CVSS reported in that database."
    para_2 += " Projects may instead calculate the severity themselves using the latest version of " + Link(("CVSS", "https://www.first.org/cvss/")) + " at the time of the vulnerability disclosure,"
    para_2 += " if the calculation inputs are publicly revealed once the vulnerability is publicly known."
    para_2 += " Note: this means that users might be left vulnerable to all attackers worldwide for up to 60 days."
    para_2 += " This criterion is often much easier to meet than what Google recommends in " + Link(("Rebooting responsible disclosure", "https://security.googleblog.com/2010/07/rebooting-responsible-disclosure-focus.html")) + ","
    para_2 += " because Google recommends that the 60-day period start when the project is notified _even_ if the report is not public."
    para_2 += " Also note that this badge criterion, like other criteria, applies to the individual project."
    para_2 += " Some projects are part of larger umbrella organizations or larger projects, possibly in multiple layers,"
    para_2 += " and many projects feed their results to other organizations and projects as part of a potentially-complex supply chain."
    para_2 += " An individual project often cannot control the rest, but an individual project can work to release a vulnerability patch in a timely way."
    para_2 += " Therefore, we focus solely on the individual project's response time."
    para_2 += " Once a patch is available from the individual project, others can determine how to deal with the patch (e.g., they can update to the newer version or they can apply just the patch as a cherry-picked solution)."
    security += para_2
    para_3 = Paragraph()
    para_3 += "The public repositories MUST NOT leak any valid private credential (e.g., a working password or private key) that is intended to limit public access."
    security += para_3
    return security


def test_markdown_table_functionality():
    result_path = this_folder/"test_table.md"
    # Classic example- employee data
    column_names = [
        "Name", # str
        "Age", # int
        "Employee ID", # int
        "Position Type", # Literal["Full-Time", "Part-Time", "Contract"]
        "Pay Type", # Literal["Hourly", "Salary", "Commission", "Lump Sum", "Contract"]
        "Pay Rate", # float.
        "Years of Experience", # int
        "Supervisor ID" # int|None
    ]
    # interns
    minimum_wage = 7.25
    supervisor_ids = [3003, 3004, 3005, 3006, 5001]
    intern_data = [
        ["Alice", 21, 1001, "Intern", "Hourly", minimum_wage, 0, supervisor_ids[0]],
        ["Bob", 22, 1002, "Intern", "Hourly", minimum_wage, 0, supervisor_ids[0]],
        ["Charlie", 23, 1003, "Intern", "Hourly", minimum_wage, 0, supervisor_ids[0]],
        ["Diana", 24, 1004, "Intern", "Hourly", minimum_wage, 0, supervisor_ids[0]]
    ]
    # full-time employees
    full_time_base_salary = 50000
    full_time_data = [
        ["Eve", 25, 2001, "Full-Time", "Salary", 50000, 1, supervisor_ids[1]],
        ["Frank", 26, 2002, "Full-Time", "Salary", 60000, 2, supervisor_ids[1]],
        ["Grace", 27, 2003, "Full-Time", "Salary", 70000, 3, supervisor_ids[1]],
        ["Hank", 28, 2004, "Full-Time", "Salary", 80000, 4, supervisor_ids[1]]
    ]
    # contract employees
    contract_data = [
        ["Ivy", 29, 3001, "Contract", "Lump Sum", 10000, 5, supervisor_ids[2]],
        ["Jack", 30, 3002, "Contract", "Lump Sum", 20000, 6, supervisor_ids[2]]
    ]
    # part-time employees
    part_time_base_hourly = 10
    part_time_data = [
        ["Kathy", 31, 1101, "Part-Time", "Hourly", 10, 7, supervisor_ids[3]],
        ["Larry", 32, 1102, "Part-Time", "Hourly", 15, 8, supervisor_ids[3]],
        ["Molly", 33, 1103, "Part-Time", "Hourly", 20, 9, supervisor_ids[3]]
    ]
    # management / supervisor data
    supervisor_base_salary = 100000
    management_base_salary = 150000
    management_data = [
        ["Nancy", 34, supervisor_ids[0], "Intern Supervisor", "Salary", 140000, 5, supervisor_ids[4]],
        ["Oscar", 35, supervisor_ids[1], "Full-Time Supervisor", "Salary", 160000, 6, supervisor_ids[4]],
        ["Patty", 36, supervisor_ids[2], "Contract Supervisor", "Salary", 180000, 7, supervisor_ids[4]],
        ["Quincy", 37, supervisor_ids[3], "Part-Time Supervisor", "Salary", 200000, 8, supervisor_ids[4]],
        ["Randy", 38, supervisor_ids[4], "Owner", "Salary", 250000, 10, None]
    ]
    
    main_employee_table = MarkdownTable(column_names=column_names)
    main_employee_data = intern_data + full_time_data + contract_data + part_time_data + management_data
    for row in main_employee_data:
        main_employee_table.add_row(row)
        
    # Decompose main table into sub-tables as in a database schema. Minimize redundancy.
    # Employee Directory
    # Employee ID, Name, Age, Position Type, Years of Experience
    # Payroll
    # Employee ID, Pay Type, Pay Rate
    # Supervision
    # Employee ID, Supervisor ID
    
    db_columns = {
        "Employee Directory": ["Employee ID", "Name", "Age", "Position Type", "Years of Experience"],
        "Payroll": ["Employee ID", "Pay Type", "Pay Rate"],
        "Supervision": ["Employee ID", "Supervisor ID"]
    }
    def ind_from_name(name:str)->int:
        return column_names.index(name)
    db_tables = {
        table_name: MarkdownTable(column_names=columns) for table_name, columns in db_columns.items()
    }
    for table_name, table in db_tables.items():
        table_columns = db_columns[table_name]
        column_inds = [ind_from_name(col) for col in table_columns]
        skip = lambda row: any([row[ind] is None for ind in column_inds])
        for row in main_employee_data:
            if skip(row):
                continue
            table_row = [row[ind] for ind in column_inds]
            table.add_row(table_row)
    # Now, build the markdown
    test_table_md = MarkdownFile()
    test_table_md.add(Header("Employee Data", 1))
    table_sections = [
        Section(table_name, data=[table]) for table_name, table in db_tables.items()
    ]
    for section in table_sections:
        test_table_md.add(section)
    test_table_md.to_file(result_path)
    
# test_markdown_table_functionality()
supported_versions = []
versions = ["initial"]
def build_security_supported_versions()->Section:
    section = Section("Supported Versions")
    para = Paragraph("When versioning is set up, the following table will be updated to reflect the supported versions of this project.")
    section += para
    table = MarkdownTable(column_names=["Version", "Supported"])
    support_symbols = [":white_check_mark:", ":x:"]
    for version in versions:
        table.add_row([version, support_symbols[0] if version in supported_versions else support_symbols[1]])
    section += table
    return section

def build_reporting_a_vulnerability()->Section:
    section = Section("Reporting a Vulnerability")
    para = Paragraph("To report a vulnerability, please contact the project author, " + author_link() + ", or open an issue on the project's GitHub repository.")
    section += para
    return section

def build_security_md(path:Path):
    security_md = MarkdownFile()
    security_md.add(build_security_policy())
    security_md.add(build_security_supported_versions())
    security_md.add(build_reporting_a_vulnerability())
    security_md.to_file(path)
    
build_security_md(security_md_path)