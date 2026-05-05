# SPDX-FileCopyrightText: Contributors to the s4casting project
#
# SPDX-License-Identifier: MPL-2.0

import json
import pickle
import re
from pathlib import Path
from typing import TypeVar

import boto3
import httpx
import pandas as pd
import tomlkit
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class FileAccess:
    """File access abstraction for local and remote files."""

    _cache_dir: Path = Path.home() / ".cache" / "s4casting"
    _s3_client = None
    _http_client: httpx.Client = httpx.Client()

    def __init__(self, locator: str) -> None:
        """Initialize the FileAccess.

        Args:
            locator (str): File locator (local path, s3://, http://, https://).
        """
        self._bytes: bytes | None = None
        self._locator = locator.removeprefix("file://")
        self._path: Path | None = None
        self._is_remote: bool = (
            self._locator.startswith("s3://")
            or self._locator.startswith("http://")
            or self._locator.startswith("https://")
        )
        self._cache: bool = self._is_remote

    def _download_from_s3(self) -> bytes:
        """Download file from S3.

        Returns:
            bytes: File content.
        """
        if not self._s3_client:
            raise ValueError("S3 credentials not set")

        bucket, key = self._locator.removeprefix("s3://").split("/", 1)
        response = self._s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()

    def _upload_to_s3(self, data: bytes) -> None:
        """Upload file to S3.

        Args:
            data (bytes): File content to upload.
        """
        s3 = boto3.client("s3")
        bucket, key = self._locator.removeprefix("s3://").split("/", 1)
        s3.put_object(Bucket=bucket, Key=key, Body=data)

    def _download_from_http(self) -> bytes:
        """Download file from HTTP/HTTPS.

        Returns:
            bytes: File content.
        """
        response = self._http_client.get(self._locator)
        return response.content

    def _upload_to_http(self, data: bytes) -> None:
        """Upload file to HTTP/HTTPS.

        Args:
            data (bytes): File content to upload.
        """
        response = self._http_client.put(self._locator, data=data)
        response.raise_for_status()

    def _bare_path(self) -> str:
        """Get the bare path without protocol.

        Returns:
            str: Bare path.
        """
        # remove the protocol from the locator with regex
        return re.sub(r"^[a-z]+://", "", self._locator)

    def _local_path(self) -> Path:
        """Get the local path for caching.

        Returns:
            Path: Local path for caching.
        """
        if self._cache:
            return self._cache_dir / self._bare_path()

        return Path(self._locator)

    def load(self) -> bytes:
        """Load file content.

        Returns:
            bytes: File content.
        """
        if self._bytes is not None:
            return self._bytes

        if self._path is None:
            self._path = self._local_path()

        if self._path is not None and self._path.exists():
            self._bytes = self._path.read_bytes()
            return self._bytes

        if self._locator.startswith("s3://"):
            self._bytes = self._download_from_s3()
        elif self._locator.startswith("http://") or self._locator.startswith("https://"):
            self._bytes = self._download_from_http()
        else:
            self._bytes = self._load_local()
            self._cache = False

        if self._cache:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_bytes(self._bytes)

        return self._bytes

    def load_json(self) -> dict:
        """Load JSON file content.

        Returns:
            dict: Parsed JSON content.
        """
        return json.loads(self.load())

    def load_toml(self) -> dict:
        """Load TOML file content.

        Returns:
            dict: Parsed TOML content.
        """
        return tomlkit.loads(self.load())

    def load_parquet(self) -> pd.DataFrame:
        """Load Parquet file content.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        path = self._local_path()
        if not path.exists():
            self.load()
        return pd.read_parquet(path)

    def save(self, data: bytes) -> None:
        """Save file content.

        Args:
            data (bytes): File content to save.
        """
        self._bytes = data

        if self._locator.startswith("s3://"):
            self._upload_to_s3(data)
        elif self._locator.startswith("http://") or self._locator.startswith("https://"):
            self._upload_to_http(data)
        else:
            self._save_local(data)
            self._cache = False

        if self._cache:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_bytes(self._bytes)

    def save_json(self, data: dict) -> None:
        """Save JSON file content.

        Args:
            data (dict): Data to save as JSON.
        """
        self.save(json.dumps(data).encode())

    def save_toml(self, data: dict) -> None:
        """Save TOML file content.

        Args:
            data (dict): Data to save as TOML.
        """
        self.save(tomlkit.dumps(data).encode())

    def load_pydantic(self) -> T:
        """Load Pydantic model from file.

        Returns:
            T: Loaded Pydantic model instance.
        """
        return pickle.loads(self.load())

    def save_pydantic(self, data: T) -> None:
        """Save Pydantic model to file.

        Args:
            data (T): Pydantic model instance to save.
        """
        self.save(pickle.dumps(data.model_dump()))

    def as_local_path(self, create: bool = False) -> Path:
        """Get the local path of the file.

        Args:
            create (bool): Whether to create the local file if it doesn't exist.

        Returns:
            Path: Local path of the file.
        """
        path = self._local_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not create and not path.exists():
            self.load()
        return path

    def _load_local(self) -> bytes:
        """Load file content from local path.

        Returns:
            bytes: File content.
        """
        self._path = Path(self._locator)
        return self._path.read_bytes() if self._path.exists() else b""

    def _save_local(self, data: bytes) -> None:
        """Save file content to local path.

        Args:
            data (bytes): File content to save.
        """
        self._path = Path(self._locator)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_bytes(data)

    def __repr__(self) -> str:
        """String representation of the FileAccess.

        Returns:
            str: String representation.
        """
        return f"FileAccess: {self._locator}"

    @classmethod
    def set_s3_credentials(cls, access_key: str, secret_key: str) -> None:
        """Set S3 credentials for file access.

        Args:
            access_key (str): AWS access key ID.
            secret_key (str): AWS secret access key.
        """
        cls._s3_client = boto3.client("s3", aws_access_key_id=access_key, aws_secret_access_key=secret_key)

    @classmethod
    def set_http_headers(cls, headers: dict[str, str]) -> None:
        """Set HTTP headers for file access.

        Args:
            headers (dict[str, str]): HTTP headers to set.
        """
        cls._http_client.headers.update(headers)
