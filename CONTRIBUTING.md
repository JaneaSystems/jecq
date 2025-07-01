# Contributing to Jecq

We welcome all contributions to this project, provided you open an issue before submitting a pull request.

## Pull requests

Before merging, all pull requests must receive at least one approval and pass all continuous integration (CI) tests.

### C++ linting

Our GitHub Actions workflow automatically runs a linting check on all C++ files in a pull request. A passing check is required to merge your contribution.

We strongly recommend using pre-commit hooks to find and fix issues on your local machine before you push your code. This saves time and helps ensure your pull request passes the check.

#### 1. Set up the pre-commit hook
This is a two-step process. First, you'll install the tool on your system. Then, you'll activate it specifically for your local copy of the Jecq repository.

Install the pre-commit package globally:

```sh
pip install pre-commit
```

Activate the hook within the Jecq repository:

From the root directory of your local project folder (the one containing this CONTRIBUTING.md file), run the following command:

```sh
pre-commit install
```

Once activated, the linter will be triggered to run on your changed files every time you use the git commit command.

#### 2. Run the linter manually (Optional)
You can also run the linter on all files in the project at any time. This is useful for checking the entire codebase at once.

```sh
pre-commit run --all-files
```

### Building the Project

A pull request can only be merged if the build passes CI checks. Our CI pipeline uses Docker to create a consistent and reproducible build environment.

You can replicate this build process on your local machine to test for any issues before pushing your changes.

#### Building with Docker (Recommended)

To run the build locally on Linux:

```sh
docker build --platform=linux/amd64 -t jecq-demo -f Dockerfile.linux .
```

Command breakdown:

```sh
--platform=linux/amd64: Specifies the target architecture, matching our Linux-based CI runner.
-t jecq-demo: Assigns the name (tag) jecq-demo to the built image for easy reference.
-f Dockerfile.linux: Points to the specific Dockerfile to use for the Linux build.
.: Tells Docker to use the current directory as the build context.
```

#### Building manually
For instructions on building the project manually, please refer to the INSTALL.md file.

## License

By contributing to the Jecq project, you agree that your contributions will be licensed under the project's license, which you can read in the [LICENSE](LICENSE) file.
