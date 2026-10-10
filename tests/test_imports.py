import subprocess
import sys
import unittest


class ImportTests(unittest.TestCase):
    def test_import_senko_does_not_import_colour(self):
        code = (
            "import sys, importlib.util, senko; "
            "assert 'colour' not in sys.modules; "
            "importlib.util.find_spec('matplotlib')"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
