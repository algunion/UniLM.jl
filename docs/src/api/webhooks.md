# [Webhooks API](@id webhooks_api)

Verify inbound webhook signatures (Standard Webhooks / HMAC-SHA256) and parse
events into typed values — an inbound utility with no provider endpoint, its HMAC built
on the `SHA` standard library —, and manage a project's webhook endpoints at OpenAI
(the `:webhooks` capability). OpenAI webhooks.

## Types and Constants

```@docs
WebhookEvent
WEBHOOK_EVENTS
WebhookEndpoint
WebhookEndpointList
WebhookEndpointTest
```

## Functions

```@docs
verify_webhook
parse_webhook
```

## Endpoint management

```@docs
create_webhook_endpoint
list_webhook_endpoints
retrieve_webhook_endpoint
update_webhook_endpoint
delete_webhook_endpoint
rotate_webhook_endpoint_secret
test_webhook_endpoint
list_webhook_event_types
```

## Result Types

```@docs
WebhookEndpointSuccess
WebhookEndpointListSuccess
WebhookEndpointDeleteSuccess
WebhookTestSuccess
WebhookEventTypesSuccess
WebhookEndpointFailure
WebhookEndpointCallError
```

## Usage

```julia
# Verify an inbound webhook, then parse it into a typed event
if verify_webhook(payload, headers, ENV["OPENAI_WEBHOOK_SECRET"])
    event = parse_webhook(payload)
    event.type in WEBHOOK_EVENTS && println("Event: ", event.type, " (", event.id, ")")   # no eval.run.* or video.*
end

# Make an endpoint, keep its signing secret, send it a sample event, then delete it
made = create_webhook_endpoint(name="batches", url="https://example.com/openai-webhooks",
                               event_types=["batch.completed", "batch.failed"])
made isa WebhookEndpointSuccess || error("the endpoint was not made: $made")
secret = made.response.signing_secret             # returned only here and by rotate_webhook_endpoint_secret
test_webhook_endpoint(made.response.id, "batch.completed")   # WebhookTestSuccess: response.status_code
delete_webhook_endpoint(made.response.id)
```

`WEBHOOK_EVENTS` leaves out the `eval.run.*` events, which OpenAI sends until its Evals API
shuts down on November 30, 2026, and `video.completed` and `video.failed`, the events of its Videos
API, which shut down on September 24, 2026
([deprecations](https://developers.openai.com/api/docs/deprecations)), though
[`list_webhook_event_types`](@ref) still returned them on November 16, 2026: the filter above
drops them, so match `event.type` yourself if you still run evals or receive video events.
