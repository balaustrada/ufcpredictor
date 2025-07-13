import os
import shutil
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from ufcpredictor.app import (  # Replace 'your_module_name' with the actual module name
    get_args,
    main,
)

THIS_DIR = Path(__file__).parent


class TestApp(unittest.TestCase):

    @patch("ufcpredictor.app.argparse.ArgumentParser.parse_args")
    def test_get_args(self, mock_parse_args):
        # Mock the return value of the parse_args method
        mock_parse_args.return_value = MagicMock(
            log_level="INFO",
            server_name="127.0.0.1",
            download_dataset=True,
            data_folder=Path("/path/to/data"),
            model_path=Path("/path/to/model"),
            port=7860,
        )

        args = get_args()
        self.assertEqual(args.log_level, "INFO")
        self.assertEqual(args.server_name, "127.0.0.1")
        self.assertTrue(args.download_dataset)
        self.assertEqual(args.data_folder, Path("/path/to/data"))
        self.assertEqual(args.model_path, Path("/path/to/model"))
        self.assertEqual(args.port, 7860)

    @patch("gradio.Blocks")  # Patch gr.Blocks
    @patch("gradio.Button")  # Patch gr.Button
    @patch("gradio.DateTime")  # Patch gr.DateTime
    @patch("gradio.Dropdown")  # Patch gr.Dropdown
    @patch("gradio.Number")  # Patch gr.Number
    def test_main_function(
        self,
        mock_Number,
        mock_Dropdown,
        mock_DateTime,
        mock_Button,
        mock_Blocks,
    ):
        config_file = THIS_DIR / "test_files" / "config_simple.yaml"
        # copy file into /tmp and edit it to modify lines

        temp_config_file = Path("/tmp/config_simple.yaml")
        shutil.copy(config_file, temp_config_file)

        # use read the file as read fieldata = file.read() then use
        # .replace(pattern, replacement) then save
        with open(temp_config_file, "r") as file:
            filedata = file.read()

        # Replace the data_folder line
        filedata = filedata.replace("data_folder_replace", f"{THIS_DIR / 'test_files'}")

        with open(temp_config_file, "w") as file:
            file.write(filedata)

        with patch("ufcpredictor.app.get_args") as mock_get_args:
            mock_get_args.return_value = MagicMock(
                config_path=temp_config_file,
                log_level="INFO",
                download_dataset=True,
                data_folder=Path(THIS_DIR / "test_files"),
                model_path=Path(THIS_DIR.parents[1] / "models" / "model.pth"),
                server_name="127.0.0.1",
                port=7860,
            )

            # Mock environment variable
            os.environ["DATASET_TOKEN"] = "fake_token"

            # Mock Gradio components
            mock_Block_instance = MagicMock()
            mock_Blocks.return_value = mock_Block_instance

            mock_Button_instance = MagicMock()
            mock_Button.return_value = mock_Button_instance

            mock_DateTime_instance = MagicMock()
            mock_DateTime.return_value = mock_DateTime_instance
            mock_DateTime_instance.value = datetime(2024, 10, 1).timestamp()

            mock_Dropdown_instance = MagicMock()
            mock_Dropdown.return_value = mock_Dropdown_instance
            mock_Dropdown_instance.value = "16c353563ae4"

            mock_Number_instance = MagicMock()
            mock_Number.return_value = mock_Number_instance
            mock_Number_instance.value = 100

            result_container = []

            def click_side_effect(function, inputs, outputs):
                result = function(
                    "d64dd9027fff 472ddc3ed02e",
                    "8bc056786ba3 fa6d7f3ba5e4",
                    datetime(2024, 10, 1).timestamp(),
                    -188,
                    188,
                )
                result_container.append(result)
                return result

            # Mock the side effect of the button click
            mock_Button_instance.click.side_effect = click_side_effect

            # Call main
            main()

            patch_ = result_container[0].axes[0].patches[0]

            self.assertAlmostEqual(0, patch_.get_x())
            self.assertAlmostEqual(0.039, patch_.get_width(), places=4)
            self.assertAlmostEqual(-0.35, patch_.get_y())
            self.assertAlmostEqual(0.7, patch_.get_height())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
