# Installation instructions

## Dependencies

The following dependencies are required for this project:

1. bmipy
2. matplotlib
3. numpy

## Installation

To install this project, run the following command in the project directory:

```python
pip install .
```

## Usage

Using this project requires a working installation of [Ngen](https://github.com/NOAA-OWP/ngen), or a working dev container with Ngen installed, such as through [NGIAB-CloudInfra](https://github.com/CIROH-UA/NGIAB-CloudInfra).

Once you have a working installation of Ngen, you can incorporate the included `formulation.json` into an Ngen realization file.

The last step is to ensure that this project is installed in the same environment as Ngen, such that it can be accessed by Ngen.
