# Testing

## Strategy
- Unit tests for core logic
- Integration tests with fixture vaults
- Golden snapshot comparison

## Idempotence Rule

~~~text
build(build(vault)) == build(vault)
~~~
