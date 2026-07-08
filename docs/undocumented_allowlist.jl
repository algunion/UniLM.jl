# Known-undocumented EXPORTED symbols — the explicit, falsifiable debt ledger.
# The doc-coverage gate (docs/doc_coverage.jl) fails when an exported symbol is
# neither in an @docs block nor listed here. Entries may only be REMOVED (by
# documenting the symbol) — never silently added. Seeded 2026-07-08.
const KNOWN_UNDOCUMENTED = Set([
    "ApplyPatchTool",
    "ComputerTool",
    "CustomTool",
    "InvalidConversationError",
    "LocalShellTool",
    "ShellTool",
    "apply_patch_tool",
    "computer_tool",
    "custom_tool",
    "incomplete_details",
    "local_shell",
    "mcp_approval_response",
    "response_status",
    "shell",
    "usage_details",
])
