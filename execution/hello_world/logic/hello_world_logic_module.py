def hello_world_prepender(s: str) -> str:
    """Prepend 'Hello World' to each line in the input string"""

    return "\n".join(["Hello World: " + line for line in s.split("\n")])
