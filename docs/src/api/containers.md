# [Containers API](@id containers_api)

Create, retrieve, list, and delete code-interpreter containers, and add files to
them. Containers provide sandboxed compute for the Responses code-interpreter
tool and expire after idle. OpenAI only.

## Objects

```@docs
ContainerObject
ContainerList
```

## Result Types

```@docs
ContainerSuccess
ContainerListSuccess
ContainerDeleteSuccess
ContainerFailure
ContainerCallError
```

## Request Functions

```@docs
create_container
retrieve_container
list_containers
delete_container
add_container_file
```

## Usage

```julia
# Create a container (optionally seeding it with uploaded file ids)
result = create_container(name="my-sandbox")
result isa ContainerSuccess && println("Created: ", result.response.id)

# Retrieve, list, add a file, delete
container = retrieve_container("cntr-abc123")
containers = list_containers(limit=20)
add_container_file("cntr-abc123", "data.csv")
delete_container("cntr-abc123")
```
