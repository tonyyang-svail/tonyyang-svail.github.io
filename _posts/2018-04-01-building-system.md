---
layout: post
title:  "Building System on C++"
comments: true
---

Despite different characteristics among programming languages, compiled or scripted, static typing or dynamic typing, the large scale software project written in a language is essentially a tree of files.

When coding in C/C++, and many other compilation languages, the following things are done separately

- Building System: a practical building system (e.g. Makefile, CMake, Bazel) does not come with the language itself, one need to install one. The extra code has to be written.

- Namespace(C++) is hardcoded in the source code, which may or may not be the same as the location of the file.

- Implementation (.c) and interface (.h) are separated. It increases the number of files in the directory Header files inclusion increases the file size, which later on slows down the compilation.

- The linkage is separated from the compilation. Even though a file passed the compilation, it can still fail on the linkage. And most of the case, `#include "A"` in the source usually implies linking to libA in the building system.

When working on a moderate size project, these complexities and repetitiveness can really slow the progress down.

For example, consider you want to move file A from folder B to folder C. As a prudent engineer, you need to

- Modify A's building system, namespace, and source:
  1. Move both the source and the header to folder C
  1. Modify the namespace from B to C
  1. Modify the makefile, also check if there is a circular dependency
- Fix other modules depends on A:
  1. Modify the header files from B/A.h to C/A.h
  1. In every use of A, modify B::A to C::A
  1. Modify the makefile on target that relies on A, also check if there is a circular dependency

We found a lot of duplications. For example

  - Source and Header could be merged
  - Namespace and file location could be merged
  - Compilation and Linking could be merge

And this is exactly what more modern languages like Python, Java and Go did.
