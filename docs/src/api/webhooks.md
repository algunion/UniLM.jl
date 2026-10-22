# [Webhooks API](@id webhooks_api)

Verify inbound webhook signatures (Standard Webhooks / HMAC-SHA256) and parse
events into typed values. This is an inbound utility with no provider endpoint —
HMAC is built on the `SHA` standard library. OpenAI webhooks.

## Types and Constants

```@docs
WebhookEvent
WEBHOOK_EVENTS
```

## Functions

```@docs
verify_webhook
parse_webhook
```

## Usage

```julia
# Verify an inbound webhook, then parse it into a typed event
if verify_webhook(payload, headers, ENV["OPENAI_WEBHOOK_SECRET"])
    event = parse_webhook(payload)
    event.type in WEBHOOK_EVENTS && println("Event: ", event.type, " (", event.id, ")")
end
```

`WEBHOOK_EVENTS` leaves out the `eval.run.*` events, which OpenAI sends until its Evals API
shuts down on November 30, 2026: the filter above drops them, so match `event.type` yourself if
you still run evals.
