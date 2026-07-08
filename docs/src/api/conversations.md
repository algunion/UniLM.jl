# [Conversations API](@id conversations_api)

Create, retrieve, update, and delete durable, server-side conversations, and
manage their items. A conversation `id` feeds the Responses API for multi-turn
exchanges without resending history. OpenAI only.

## Parsed Objects

```@docs
ConversationObject
ConversationItem
ConversationItemList
```

## Result Types

```@docs
ConversationSuccess
ConversationItemListSuccess
ConversationItemSuccess
ConversationDeleteSuccess
ConversationFailure
ConversationCallError
```

## Request Functions

```@docs
create_conversation
retrieve_conversation
update_conversation
delete_conversation
add_conversation_items
list_conversation_items
delete_conversation_item
```

## Accessors

```@docs
conversation_id
```

## Usage

```julia
# Create a durable, server-side conversation
result = create_conversation(metadata=Dict("topic" => "support"))
result isa ConversationSuccess && println("Created: ", conversation_id(result.response))

# Retrieve, then update its metadata
retrieve_conversation("conv_abc123")
update_conversation("conv_abc123", Dict("status" => "closed"))

# Add items, list them, then clean up
add_conversation_items("conv_abc123", [InputMessage(role="user", content="Hello")])
items = list_conversation_items("conv_abc123", limit=20)
delete_conversation_item("conv_abc123", "msg_xyz")
delete_conversation("conv_abc123")
```
