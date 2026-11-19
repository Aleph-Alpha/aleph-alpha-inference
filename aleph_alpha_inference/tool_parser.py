# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 Aleph Alpha GmbH
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser


# Kolibri 1 currently shares the Hermes `<tool_call>...</tool_call>` format.
class Kolibri1ToolParser(Hermes2ProToolParser):
    pass
