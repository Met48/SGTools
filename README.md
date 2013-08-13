## Skullgirls Tools

### Introduction

Currently one tool, a sprite decoder. Skullgirls sprites are tile-based and have a .spr.msb file describing the layout.
This command-line tool can generate the full frames of animation, with transparency. It can also generate the sprites
the rely on a palette, such as character sprites.

### Requirements

- docopt: command-line interface
- numpy: image processing
- PIL: image loading and saving

### License

The code is under the MIT license.
