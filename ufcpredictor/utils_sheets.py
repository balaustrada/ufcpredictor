"""
This module provides functionality to read data from Google Sheets using the Google
Sheets API.
It includes a class `SheetsReader` for handling authentication and reading data, and a
function `read_fights_sheet` to read specific fight data from a given spreadsheet.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

if TYPE_CHECKING:  # pragma: no cover
    from typing import Callable, Dict, Optional

    from numpy.typing import NDArray


class SheetsReader:
    """
    A class to read data from Google Sheets using the Google Sheets API.

    Attributes:
        creds_file (Path): Path to the credentials file for Google Sheets API.
        scopes (list[str]): List of scopes for the API access.
        creds (Credentials | None): Credentials object for authentication.
        service (build | None): Google Sheets API service object.
    """

    def __init__(self, creds_file: Path, scopes: list[str]):
        """
        Initializes the SheetsReader with the given credentials file and scopes.

        Args:
            creds_file (Path): Path to the credentials file.
            scopes (list[str]): List of scopes for the API access.
        """
        self.creds_file = creds_file
        self.scopes = scopes
        self.creds: Credentials | None = None
        self.service: build | None = None

    def authenticate(self) -> None:
        """Authenticates the user and initializes the service."""
        self.creds = Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            self.creds_file,
            scopes=self.scopes,
        )

        self.service = build("sheets", "v4", credentials=self.creds)

    def read_sheet(self, spreadsheet_id: str, range_name: str) -> list[list[str]]:
        """Reads data from a specified sheet and range."""
        if not self.service:
            raise Exception("Service not initialized. Call authenticate() first.")

        try:
            sheet = self.service.spreadsheets()
            result = (
                sheet.values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )
            return result.get("values", [])
        except HttpError as err:
            print(f"An error occurred: {err}")
            return []


def read_fights_sheet(
    spreadsheet_id: str,
    creds_file: Path,
    fields_to_read: list[str],
    dtypes_: Optional[list[type]] = None,
) -> list[NDArray]:
    """Reads the fights sheet from the specified Google Sheets document.

    Args:
        spreadsheet_id: The ID of the Google Sheets document.
        creds_file: The path to the credentials file.
        fields_to_read: The list of fields (columns) to read from the sheet.
        dtypes_: List with the numpy types to apply to each column.

    Returns:
        List of numpy arrays with the column informations.
    """
    reader = SheetsReader(
        creds_file=creds_file,
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
    )

    reader.authenticate()
    range_name = "Fights!A1:M"

    data = reader.read_sheet(spreadsheet_id, range_name)

    if not data:
        raise ValueError("No data found in the specified range.")

    headers = data[0]
    columns = list(map(list, zip(*data[1:])))

    # Check all columns have the same length
    # raise error
    if not all(len(col) == len(columns[0]) for col in columns):
        raise ValueError("All columns must have the same length.")

    output_arrays = []
    for i, col_name in enumerate(fields_to_read):
        if col_name in headers:
            index = headers.index(col_name)
            output_arrays.append(np.asarray(columns[index]))
        else:
            raise ValueError(f"Column '{col_name}' not found in the sheet.")

    if dtypes_:
        for i, dtype_ in enumerate(dtypes_):
            output_arrays[i] = output_arrays[i].astype(dtype_)

    return output_arrays
