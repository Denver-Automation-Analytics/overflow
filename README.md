# Overflow

Developed by Michael Baker International - Tools for generating hydrologic data on massive grids.

## How to contribute

This repository contains the source code for the various tools available through the `overflow` package and cli. This repository provides a Python API and CLI to each tool.

## Prerequisites

Before you can start developing, please make sure you have the following software installed on your machine:

- Docker (for using the development container)
- Visual Studio Code (recommended editor with Remote - Containers extension installed)

## Dev-Container Setup

This project uses a dev-container for development. For more info, read the documentation [here](https://code.visualstudio.com/docs/devcontainers/containers)

## Python APIs

The Python API is accessed through the `overflow` Python package. You will find the following in the `src/` folder related to the Python API:

- `overflow` Python package
  - This is the main package in this repository
  - This package contains the Python APIs for each tool

## CLIs

The CLI for each tool is defined in `src/overflow_cli.py`. It uses the `click` Python package. Click documentation is available [here](https://click.palletsprojects.com/en). Each new tool should have a CLI developed for it.

## Authentication

No authentication with any services other than GitHub is required to develop on this repository.

## Running The Tests

After building the dev-container, all dependencies should be installed and you should be ready to start developing. The first thing you should do is verify the all the tests can run successfully. To do this, click the tests button on the extensions panel of VSCode. You may need to configure the tests with the GUI. To do this, select _Configure Tests_ and choose _pytest_ as the testing framework and `tests` as the testing directory.

Once the tests are loaded, you can run them with the play button in the tests extension.

Alternatively, you can run the following command from the root directory to run all tests

```bash
pytest
```

## Linting

This project includes a linter for Python called `pylint`. Your source code will be linted in the editor. To run the linter manually, run the following command from the root directory

```bash
pylint
```

## CI/CD

This project incorporates a CI/CD pipeline to test and deploy updates. Any merge to the staging branch will build and push an update to the staging `overflow` environment. Any merge to the production branch will build and push an update to the production `overflow` environment.

Additionally, all pull requests will automatically run the tests and lint your code. If any tests or linting fails, the pull request cannot be merged.

## Formatting Guidelines

This project uses `black` as the Python formatter. Before you merge your code into staging or main, you should make sure all your code is formatted by running the following command from the root of the project.

```ps
black .
```

## Running tools in development

### Notes on running command line tools during development

Click has been selected as the command line wrapper for our projects. To run tools parameterized through click,  
Follow this example in the terminal:

```ps
python <path_to_python_file> <name_of_function> --<function_parameter_name> <function_parameter_value>
```

To get the help messages:

```ps
python <path_to_python_file> --help
```

### Notes on running command line tools once image is built

If you want to run the container, build or pull it, then:

```ps
docker run -v <bind mount> <image name> /dist/overflow_cli <command_name> --<command_arg_name> <command_arg_value> ...
```

Or from inside the container, run the tools as follows:

```ps
/dist/overflow_cli <command_name> --<command_arg_name> <command_arg_value> ...
```

For help on a particular command

```ps
/dist/overflow_cli <command_name> --help
```

For a list of all available commands

```ps
/dist/overflow_cli --help
```

---

## Building Dev Image

To test this image without pushing to staging or production branches, you can build it locally on Windows using the build-dev-image.ps1 script. First, you may need to set the execution policy

In powershell:

1. Set Execution Policy (If Needed): If you encounter an issue running scripts due to the execution policy, you can temporarily set it to unrestricted. Run the following command:

   ```ps
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Unrestricted
   ```

2. Run the Script: Once you are in the correct directory, run your PowerShell script using the following command:

   ```ps
   .\build-dev-image.ps1
   ```
