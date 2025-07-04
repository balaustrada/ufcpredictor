"""
This module provides functionality to read data from Google Sheets using the Google 
Sheets API.
It includes a class `SheetsReader` for handling authentication and reading data, and a 
function `read_fights_sheet` to read specific fight data from a given spreadsheet.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

if TYPE_CHECKING:  # pragma: no cover
    from typing import Callable, Dict, Optional


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
        try:
            self.creds = Credentials.from_authorized_user_file( # type: ignore[no-untyped-call]
                self.creds_file, self.scopes
            )
        except Exception as e:
            print(f"Error loading credentials: {e}")
            flow = InstalledAppFlow.from_client_secrets_file(
                self.creds_file, self.scopes
            )
            self.creds = flow.run_local_server(port=0)

        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request()) # type: ignore[no-untyped-call]
            else:
                raise Exception("Invalid credentials")

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
    transformations_to_apply: Optional[list[Callable]] = None,
) -> Dict[str, list]:
    """Reads the fights sheet from the specified Google Sheets document.

    Args:
        spreadsheet_id: The ID of the Google Sheets document.
        creds_file: The path to the credentials file.
        fields_to_read: The list of fields (columns) to read from the sheet.
        transformations_to_apply: A list of transformation functions to apply to each 
            column.

    Returns:
        Dict[str, list]: A dictionary containing the requested fields and their values.
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

    data_dict = {}
    for i, col_name in enumerate(fields_to_read):
        if col_name in headers:
            index = headers.index(col_name)
            data_dict[col_name] = columns[index]
        else:
            raise ValueError(f"Column '{col_name}' not found in the sheet.")

    if transformations_to_apply:
        # check length of transformations_to_apply matches length of columns_to_read
        if len(transformations_to_apply) != len(fields_to_read):
            raise ValueError(
                "Number of transformations does not match number of columns to read."
            )

        for i, (header, column) in enumerate(data_dict.items()):
            data_dict[header] = [
                transformations_to_apply[i](val) if val else None for val in column
            ]

    return data_dict
