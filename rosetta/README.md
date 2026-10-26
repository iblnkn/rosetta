# rosetta

The core Rosetta package: the contract loader, the frame pipeline, the episode
recorder and policy runner nodes, and the bag porter (`rosetta_port`).

Framework adapters and apps such as [`rosetta_hil`](../rosetta_hil) build on
its public Python API, listed on the
[Python API reference](https://iblnkn.github.io/rosetta/reference/python-api.html)
page.

## Build

```bash
colcon build --packages-up-to rosetta
source install/setup.bash
```

## Documentation

Full Rosetta documentation: **https://iblnkn.github.io/rosetta/**

## License

Apache-2.0
