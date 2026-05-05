# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import sys
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    CliSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)

from s4casting.core.config import Configuration


def print_help():
    """Print help message for CLI usage."""
    print("Usage: torchrun scripts/train.py <config.toml> <cli_args>")  # noqa: T201


def get_configuration() -> Configuration:
    """Get configuration from CLI arguments and configuration file.

    Returns:
        Configuration: The instantiated configuration object.
    """
    if len(sys.argv) < 2 or sys.argv[1].lower() in ["-h", "--help"]:
        print_help()
        sys.exit(0)

    if not Path(sys.argv[1]).exists():
        print("Please provide a configuration file.")  # noqa: T201
        print_help()
        sys.exit(1)

    class InstantiatedConfiguration(Configuration):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,  # noqa
            dotenv_settings: PydanticBaseSettingsSource,  # noqa
            file_secret_settings: PydanticBaseSettingsSource,  # noqa
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            """Customize settings sources to include CLI and TOML file.

            Args:
                settings_cls (type[BaseSettings]): Settings class.
                init_settings (PydanticBaseSettingsSource): Initial settings source.
                env_settings (PydanticBaseSettingsSource): Environment settings source.
                dotenv_settings (PydanticBaseSettingsSource): Dotenv settings source.
                file_secret_settings (PydanticBaseSettingsSource): File secret settings source.

            Returns:
                tuple[PydanticBaseSettingsSource, ...]: Customized settings sources.
            """
            return (
                init_settings,
                CliSettingsSource(settings_cls, cli_prog_name="s4casting", cli_parse_args=sys.argv[2:]),
                EnvSettingsSource(settings_cls, env_prefix="S4_", env_nested_delimiter="__", case_sensitive=False),
                TomlConfigSettingsSource(settings_cls, toml_file=sys.argv[1]),
            )

    return InstantiatedConfiguration()  # type: ignore[call-arg]
