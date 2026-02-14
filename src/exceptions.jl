"""
    InvalidConversationError <: Exception

Thrown when a conversation violates structural rules (e.g., missing system message,
consecutive messages from the same role).

# Fields
- `reason::String`: Human-readable explanation of why the conversation is invalid.
"""
struct InvalidConversationError <: Exception
    reason::String
end