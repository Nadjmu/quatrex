# Command Line Interface

The usual way of launching `quatrex` is via its command line interface
(CLI). There is one main command, `quatrex run`, which is used to run a
simulation. It takes as an argument the path to a
[TOML](https://toml.io/) file containing the [simulation parameters](parameters/index.md).

## :octicons-command-palette-24: `quatrex`

```bash
quatrex [OPTIONS] COMMAND [ARGS]...
```

| Option      | Description                 |
| ----------- | --------------------------- |
| `--version` | Print the version and exit. |
| `--help`    | Show a help message.        |

## :octicons-command-palette-24: `quatrex mesh`

```bash
quatrex mesh [OPTIONS] CONFIG
```

Generates and visualizes the device's electrostatics mesh based on the
provided configuration. This has to be invoked before running the
simulation.


| Argument | Description                                  |
| -------- | -------------------------------------------- |
| `config` | Path to the quatrex TOML configuration file. |


| Option         | Description                          |
| -------------- | ------------------------------------ |
| `--off-screen` | Whether to use off-screen rendering. A visualization of the mesh will be generated without displaying it on the screen. |
| `--help`       | Show a help message.                 |

## :octicons-command-palette-24: `quatrex run`

```bash
quatrex run [OPTIONS] [CONFIG]
```

Run a simulation with the given configuration file.

| Argument | Description                                                                                                                                                                                                                                    |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config` | Path to the quatrex TOML configuration file. This can also be a path to a folder containing the configuration file (`quatrex_config.toml`). If not given, `quatrex` will look for a file named `quatrex_config.toml` in the current directory. |

| Option                                            | Description                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `--abort-on-exception`/ `--no-abort-on-exception` | Force abort the entire MPI environment on an unhandled exception to prevent hanging processes. |
| `--help`                                          | Show a help message.                                                                           |

