# Agent Based Model Simulation using a Social Force model
To view the paper we have written on this subject: _paper link_

# How to run
* Run `pip install -r requirements.txt` to install the necessary Python modules.
* Run `main.py` to run your simulation from that moment on.

# How to write an .absml file
## Introduction
An .absml file is practically a glorified .toml file we use to parse your crowd simulation logic. The .absml file represents a tree structure which may be cyclic, which means that all objects in your file are linked in one way or another. There are different objects one can use in their simulation. Note that object and node are used synonymously throughout this file.

## Basic Syntax
* To initialize an object, use the square brackets `[]` just like you would use in a .toml file.
* The object names must start with the class name, but can be followed by any characters (e.g. `spawnerNumberOne` is allowed but `firstSpawner` is not).
* The `[global]` node is used for miscellaneous global variables.
* The master node (the entry point of the program) is denoted by a `!` suffix, and all programs must have exactly one master node.
The following code creates a spawner that redirects the pedestrians to a waiting area:

```toml
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

## Spawner Object
