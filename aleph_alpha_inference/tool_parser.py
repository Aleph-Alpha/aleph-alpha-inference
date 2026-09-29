from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


# Kolibri 1 currently shares the Hermes `<tool_call>...</tool_call>` format.
class Kolibri1ToolParser(Hermes2ProToolParser):
    pass
