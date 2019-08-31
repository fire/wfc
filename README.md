# fast-wfc

An implementation of [Wave Function Collapse](https://github.com/mxgmn/WaveFunctionCollapse) with a focus on performance.
It was called fast-wfc because at the time it introduced optimizations improving the execution time by an order of magnitude.

This branch is based upon https://github.com/math-fehr/fast-wfc.

# Requirements

You need a C++-11 compatible compiler, and CMake installed.

# Install the library

```
git clone https://github.com/math-fehr/fast-wfc && cd fast-wfc/
cmake .
make install
```

will install the library `fastwfc` and `fastwfc_static` using CMake:

# Run the examples

```
# Go to install directory
cd share/fastwfc-examples
./fastwfc_demo
```

will execute WFC on the examples defined in `example/samples.xml`, and will put the results in `example/results`.

# Third-parties library

The files in `example/src/include/external/` come from:
* RapidXML [https://github.com/dwd/rapidxml](https://github.com/dwd/rapidxml)
* stb Library [https://github.com/nothings/stb](https://github.com/nothings/stb)

# Image samples

The image samples come from [https://github.com/mxgmn/WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse)

# Licence 

Copyright (c) 2018-2019 Mathieu Fehr and Nathanaël Courant.

MIT License, see `LICENSE` for further details.
