# cppkalman

A port of [pykalman](https://github.com/pykalman/pykalman)'s `AdditiveUnscentedKalmanFilter` to C++. The library consists of a single header file (`cppkalman.hpp`).

### Getting Started
Be sure to clone the repository recursively (or install eigen globally). Then, see `minimal.cpp` for a minimal example usage.

###### Linux (gcc/clang)
```shell
$ g++ minimal.cpp -std=c++17 -Ieigen
```

###### Windows (msvc)
*NOTE: Run this from an MSVC enabled terminal (such as `Developer Command Prompt for VS 2022`)*
```cmd
> cl minimal.cpp /std:c++17 /EHsc -Ieigen
```
