import importlib
import unittest


class CoreImportTests(unittest.TestCase):
    def test_core_namespace_imports_without_stale_exports(self) -> None:
        core = importlib.import_module("src.core")

        self.assertEqual(core.__all__, [])


if __name__ == "__main__":
    unittest.main()
