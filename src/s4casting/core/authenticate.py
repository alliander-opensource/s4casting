# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

from s4casting.core.config import AuthenticationConfiguration
from s4casting.data.files.loader import FileAccess


def configure_authentication(auth: AuthenticationConfiguration) -> None:
    """Configure authentication for file access.

    Args:
        auth (AuthenticationConfiguration): Authentication configuration.
    """
    if auth.aws_access_key_id and auth.aws_secret_access_key:
        FileAccess.set_s3_credentials(
            access_key=auth.aws_access_key_id.get_secret_value(),
            secret_key=auth.aws_secret_access_key.get_secret_value(),
        )
