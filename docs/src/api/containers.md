# [Containers API](@id containers_api)

Create, retrieve, list, and delete code-interpreter containers, and add, list,
read, download and delete the files in them. Containers provide sandboxed compute
for the Responses code-interpreter tool and expire after idle. OpenAI only.

## Objects

```@docs
ContainerObject
ContainerList
UniLM.ContainerFileObject
ContainerFileList
```

`add_container_file` and `retrieve_container_file` return a `ContainerSuccess` whose
`response` is a `ContainerFileObject` (not exported); create and retrieve return a
`ContainerObject`. `container_file_content` returns the file's bytes as a
[`FileContentSuccess`](@ref), which [`save_file_content`](@ref) writes to disk.

## Request Types

```@docs
ContainerNetworkPolicy
ContainerNetworkDisabled
ContainerNetworkAllowlist
ContainerDomainSecret
SkillReference
```

## Result Types

```@docs
ContainerSuccess
ContainerListSuccess
ContainerFileListSuccess
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
list_container_files
retrieve_container_file
container_file_content
delete_container_file
```

## Usage

```julia
# Create a container with 4 GB of memory that reaches one domain
result = create_container(name="my-sandbox", memory_limit="4g",
                          network_policy=ContainerNetworkAllowlist(["pypi.org"]))
result isa ContainerSuccess && println("Created: ", result.response.id)

# Retrieve, list by name, add files (a local path, or an uploaded file's id)
container = retrieve_container("cntr-abc123")
containers = list_containers(name="my-sandbox", order="asc")
add_container_file("cntr-abc123", "data.csv")
add_container_file("cntr-abc123"; file_id="file-abc123")

# The container's files: list, read one's record, download, delete
files = list_container_files("cntr-abc123")
retrieve_container_file("cntr-abc123", "cfile-abc123")
out = container_file_content("cntr-abc123", "cfile-abc123")
out isa FileContentSuccess && save_file_content(out, "result.csv")
delete_container_file("cntr-abc123", "cfile-abc123")
delete_container("cntr-abc123")
```
