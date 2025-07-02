from langchain_core.tools import tool
from pydantic import BaseModel, Field
from pathlib import Path
import os

# ---------------------------
# Multiply Tool
# ---------------------------

class MultiplyInput(BaseModel):
    a: float = Field(..., description="The first number.")
    b: float = Field(..., description="The second number.")

@tool(
    args_schema=MultiplyInput,
    return_direct=False,
    
)
def multiply(a: float, b: float) -> tuple[str, float]:
    """Multiply two numbers."""
    result = a * b
    return f"{a} * {b} = {result}", result


# ---------------------------
# Check Weather Tool
# ---------------------------

class WeatherInput(BaseModel):
    location: str = Field(..., description="The location to check the weather for.")

@tool(
    args_schema=WeatherInput,
    
)
def check_weather(location: str) -> str:
    
    """Return the weather forecast for the specified location."""
    return f"It's always sunny in {location}"


@tool(
    
)
def allowed_location_dir() -> str:
    """Return the absolute path of a predefined directory that the agent can access."""
    allowed_path = Path.cwd()
    if allowed_path.exists() and allowed_path.is_dir():
        return str(allowed_path.resolve())
    else:
        return "The predefined directory is invalid or inaccessible."

# ---------------------------
# List Files Tool
# ---------------------------

class ListFilesInput(BaseModel):
    directory: str = Field(
        default=".",
        description="Directory path to list files from. Leave empty to use the default allowed path."
    )
    max_depth: int = Field(
        default=1,
        ge=0,
        description="Maximum depth for recursive directory listing. 0 means only the root directory."
    )
@tool(
    args_schema=ListFilesInput,
    
)
def list_files(directory: str, max_depth: int = 1) -> str:
    """
    List all files in the specified directory as a tree up to a certain depth.
    :param directory: thư mục cần xem.
    :param max_depth: số cấp con tối đa, mặc định = 1.
    """
    directory = directory or allowed_location_dir()

    if not os.path.exists(directory):
        return f"Directory '{directory}' does not exist."
    if not os.path.isdir(directory):
        return f"'{directory}' is not a directory."

    tree_lines = []
    base_depth = directory.rstrip(os.sep).count(os.sep)

    def walk(dir_path: str, prefix: str = ""):
        current_depth = dir_path.rstrip(os.sep).count(os.sep) - base_depth
        if current_depth > max_depth:
            return

        entries = sorted(os.listdir(dir_path))
        for idx, entry in enumerate(entries):
            full_path = os.path.join(dir_path, entry)
            connector = "└── " if idx == len(entries) - 1 else "├── "
            tree_lines.append(f"{prefix}{connector}{entry}")

            # Nếu là thư mục và sâu chưa vượt max_depth
            if os.path.isdir(full_path) and current_depth < max_depth:
                extension = "    " if idx == len(entries) - 1 else "│   "
                walk(full_path, prefix + extension)

    tree_lines.append(directory.rstrip(os.sep) + os.sep)
    walk(directory)

    return "\n".join(tree_lines)


# ---------------------------
# Export list
# ---------------------------

__all__ = [
    "check_weather",
    "multiply",
    "allowed_location_dir",
    "list_files"
]