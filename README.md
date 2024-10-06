# Agent Based Model Simulation using a Social Force model
To view the paper we have written on this subject: _paper link_

# How to run
* Run `pip install -r requirements.txt` to install the necessary Python modules.
* Run `main.py` to run your simulation from that moment on.

# How to write an .absml file
## Introduction
An .absml file is practically a glorified .toml file we use to parse your crowd simulation logic. The .absml file represents a tree structure which may be cyclic, which means that all objects in your file are linked in one way or another. There are different objects one can use in their simulation. Note that object and node are used synonymously throughout this file.

## Basic Syntax
* To initialize an object, use the square brackets `[]` just like you would use in a `.toml` file.
* The object names must start with the class name, but can be followed by any characters (e.g. `spawnerNumberOne` is allowed but `firstSpawner` is not).
* The `[global]` node is used for miscellaneous global variables.
* The master node (the entry point of the program) is denoted by a `!` suffix, and all programs must have exactly one master node.
The following code creates a spawner that redirects the pedestrians to a waiting area:

```toml
[global]
target_fps = 0

["spawnerMain!"]
line = [[220, 700], [270, 700]]
wait = 0.1
limit = 100
child = "areaWaiting"

[areaWaiting]
area = [60, 100, 300, 300]
dimensions = [5, 5]
wait = "UNIFORM(2, 4)"
```

There are multiple important things to note:
* Nodes must be linked with each other with by using the `child` attribute.
* The `wait` attribute won't have any effect on the `areaWaiting` object, since it doesn't have a child node the pedestrians can go to next.
* Double quotes are used in the first node since the exclamation mark can't be parsed by `toml`.

## Spawner Object
### Brief
The place where pedestrians can spawn. Think of it like the entrance of a building.

### Attributes:
* `line`: `List[List[int], List[int]]`: the line at which the pedestrians can enter the simulation. The larger the line, the more random the entry points of the pedestrians will by.
* `wait` `float | str`: the inverse of the frequency of the spawner. If given a `float`, the number itself is used. String can be given in this format: `"UNIFORM(a, b)"`. The following functions are also supported: `GAUSS`
* `limit` `Optional[int]`: the maximum amount of pedestrians the spawner can spawn before stopping.
* `child`: `str`: the child node of the spawner, can for example be a waiting area where the pedestrians can go.

## Area Object
### Brief
An area where pedestrians can go. Think of it like the waiting area at an airport, after you have checked in.

### Attributes:
* `area`: `List[int, int, int, int]`: the rectangular area. The input must be `x`, `y`, `w`, `h`.
* `dimensions`: `List[int, int]`: the dimensions of the area where people can wait. The smaller the dimensions, the less people fit in the area (think of it as the number of chairs at a lunch area).
* `wait`: same as any `wait`
* `child`: same as any `child`