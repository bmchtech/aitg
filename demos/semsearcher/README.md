
# semsearcher

index book:
```sh
DEBUG=1 poetry run semsearcher index http://localhost:6000 ~/Downloads/ghoststory.txt ghoststory.semix
```

search book:
```sh
 DEBUG=1 poetry run semsearcher search http://localhost:6000 ghoststory.semix "What is a NGO?"
```

server ui:
```sh
poetry run semsearcherui http://localhost:6000 .
```