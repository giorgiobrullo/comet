import unittest
from unittest.mock import patch

from comet.core.models import settings
from comet.scrapers.models import ScrapeRequest
from comet.scrapers.torbox import TorboxScraper
from comet.services.torrent_manager import download_torrent


class _FailingSession:
    def get(self, *args, **kwargs):
        del args, kwargs
        raise OSError("name does not resolve")


def logged_text(*mocks):
    return " ".join(str(call) for mock in mocks for call in mock.call_args_list)


class ScraperLogSecretTests(unittest.IsolatedAsyncioTestCase):
    async def test_torbox_failure_does_not_log_the_api_key(self):
        request = ScrapeRequest(
            media_type="movie", media_id="tt123", media_only_id="tt123", title="T"
        )
        with (
            patch.object(settings, "TORBOX_API_KEY", "torbox-secret-key"),
            patch("comet.core.logger.logger.warning") as warning,
            patch("comet.core.logger.logger.log") as log,
        ):
            torrents = await TorboxScraper(None, _FailingSession()).scrape(request)

        self.assertEqual(torrents, [])
        self.assertTrue(warning.called or log.called)
        self.assertNotIn("torbox-secret-key", logged_text(warning, log))

    async def test_failed_torrent_download_does_not_log_the_query_string(self):
        url = "http://jackett:9117/dl/indexer/?jackett_apikey=jackett-secret&path=x"
        with patch("comet.services.torrent_manager.logger.debug") as debug:
            result = await download_torrent(_FailingSession(), url)

        self.assertEqual(result, (None, None, None))
        self.assertIn("http://jackett:9117/dl/indexer/", logged_text(debug))
        self.assertNotIn("jackett-secret", logged_text(debug))
