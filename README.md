# Code Sonification

This program takes a JavaScript source file as input and generates a MIDI file that "sounds like" the code. The MIDI file contains three tracks, representing code ranging from simple to complex, and can be imported into a DAW to choose any instruments you like.

## Usage

Make sure you have [Rust](https://www.rust-lang.org) installed. Then download the repository, and from your terminal, run:

```shell
cd path/to/code-sonification
cat path/to/code.js | cargo run > path/to/output.mid
```

## How it works

The program operates in four steps:

1.  Parse the JavaScript program.
2.  Traverse the syntax tree, and for each leaf statement, generate a hash and determine the complexity of the statement (ie. how many expressions it contains).
3.  Based on these hashes, determine which track to use, which chord to play, and whether to add a new sequence to or remove an existing sequence from the track. The number of notes in a sequence increases with complexity.
4.  Export the sequences as a MIDI file.
