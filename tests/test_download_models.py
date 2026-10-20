"""Base model preparation must never scatter downloaded resources outside weights/."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import download_models


class BaseDownloadLayoutTests(unittest.TestCase):
    def test_real_installer_routes_downloads_and_selected_archive_members_to_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def fake_download(url, destination):
                destination = Path(destination)
                self.assertTrue(destination.is_relative_to(root / 'weights'))
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.suffix == '.zip':
                    with zipfile.ZipFile(destination, 'w') as archive:
                        archive.writestr('models/arcface50_checkpoint.tar', b'arcface50 fixture')
                        archive.writestr('models/arcface100_checkpoint.tar', b'arcface100 fixture')
                        archive.writestr('unused.bin', b'not selected')
                else:
                    destination.write_bytes(b'public download fixture, not loaded')
                return destination
            with patch.object(download_models, 'ROOT', root), \
                 patch.object(download_models, 'download', side_effect=fake_download), \
                 patch('sys.argv', ['download_models.py']):
                download_models.main()
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['weights'])
            self.assertEqual((root / 'weights/faceshield/arcface50_checkpoint.tar').read_bytes(), b'arcface50 fixture')
            self.assertTrue((root / 'weights/ip-adapter/ip-adapter_sd15.bin').is_file())
            self.assertTrue((root / 'weights/mtcnn-attack/pnet-6b6ef92b.pth').is_file())
            self.assertFalse(list(root.rglob('unused.bin')))


if __name__ == '__main__':
    unittest.main()
