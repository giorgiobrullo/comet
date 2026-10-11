import asyncio
import unittest
from unittest.mock import patch

from comet.core.models import AppSettings
from comet.core.scrape import ScrapeContext
from comet.services import orchestration
from comet.services.orchestration import TorrentManager, scraper_manager, settings


def make_manager():
    return TorrentManager(
        media_type="movie",
        media_full_id="tt123",
        media_only_id="tt123",
        title="Title",
        year=2024,
        year_end=None,
        season=None,
        episode=None,
        aliases={},
        remove_adult_content=False,
    )


def scraped(info_hash):
    return {
        "infoHash": info_hash,
        "fileIndex": None,
        "title": f"Title.2024.1080p.{info_hash}",
        "seeders": 1,
        "size": 1,
        "tracker": "tracker",
        "sources": [],
        "parsed": object(),
    }


class LiveResponseDeadlineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.release_slow = asyncio.Event()
        self.slow_done = asyncio.Event()
        self.cached = []

    async def scrapers(self, request):
        del request
        yield "Fast", [scraped("fast")], 0.1
        await self.release_slow.wait()
        yield "Slow", [scraped("slow")], 36.0

    async def filter_manager(self, manager, scraper_name, torrents, response_time):
        del scraper_name, response_time
        manager.ready_to_cache.extend(torrents)

    def patches(self, manager, deadline):
        async def cache_torrents(torrents=None):
            batch = manager.ready_to_cache if torrents is None else torrents
            self.cached.append([torrent["infoHash"] for torrent in batch])

        async def filter_manager(scraper_name, torrents, response_time=None):
            await self.filter_manager(manager, scraper_name, torrents, response_time)

        return (
            patch.object(settings, "LIVE_SCRAPE_RESPONSE_DEADLINE", deadline),
            patch.object(scraper_manager, "scrape_all", new=self.scrapers),
            patch.object(manager, "filter_manager", new=filter_manager),
            patch.object(manager, "cache_torrents", new=cache_torrents),
            patch.object(manager, "_matches_requested_scope", return_value=True),
        )

    async def test_answers_at_deadline_and_caches_late_results(self):
        manager = make_manager()
        late = []

        async def on_late_torrents(torrents):
            late.append(sorted(torrents))

        p1, p2, p3, p4, p5 = self.patches(manager, 0.05)
        with p1, p2, p3, p4, p5:
            await manager.scrape_torrents(ScrapeContext.LIVE, on_late_torrents)

            self.assertEqual(list(manager.torrents), ["fast"])
            self.assertEqual(self.cached, [["fast"]])
            self.assertEqual(len(orchestration._late_collections), 1)

            self.release_slow.set()
            await asyncio.gather(*orchestration._late_collections)

        self.assertEqual(self.cached, [["fast"], ["slow"]])
        self.assertEqual(late, [["slow"]])
        self.assertEqual(list(manager.torrents), ["fast"])
        self.assertEqual(orchestration._late_collections, set())

    async def test_waits_for_all_scrapers_that_finish_before_the_deadline(self):
        manager = make_manager()
        late = []

        async def on_late_torrents(torrents):
            late.append(torrents)

        self.release_slow.set()
        p1, p2, p3, p4, p5 = self.patches(manager, 5)
        with p1, p2, p3, p4, p5:
            await manager.scrape_torrents(ScrapeContext.LIVE, on_late_torrents)

        self.assertEqual(sorted(manager.torrents), ["fast", "slow"])
        self.assertEqual(self.cached, [["fast", "slow"]])
        self.assertEqual(late, [])
        self.assertEqual(orchestration._late_collections, set())

    async def test_background_scrapes_ignore_the_deadline(self):
        manager = make_manager()
        p1, p2, p3, p4, p5 = self.patches(manager, 0.01)
        with p1, p2, p3, p4, p5:
            scrape = asyncio.create_task(
                manager.scrape_torrents(ScrapeContext.BACKGROUND)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(scrape.done())
            self.release_slow.set()
            await scrape

        self.assertEqual(sorted(manager.torrents), ["fast", "slow"])
        self.assertEqual(self.cached, [["fast", "slow"]])

    async def test_without_a_deadline_the_response_waits_for_every_scraper(self):
        manager = make_manager()
        p1, p2, p3, p4, p5 = self.patches(manager, None)
        with p1, p2, p3, p4, p5:
            scrape = asyncio.create_task(manager.scrape_torrents(ScrapeContext.LIVE))
            await asyncio.sleep(0.05)
            self.assertFalse(scrape.done())
            self.release_slow.set()
            await scrape

        self.assertEqual(sorted(manager.torrents), ["fast", "slow"])
        self.assertEqual(self.cached, [["fast", "slow"]])

    async def test_a_cancelled_request_still_fills_the_cache(self):
        manager = make_manager()
        p1, p2, p3, p4, p5 = self.patches(manager, 30)
        with p1, p2, p3, p4, p5:
            scrape = asyncio.create_task(manager.scrape_torrents(ScrapeContext.LIVE))
            await asyncio.sleep(0.01)
            scrape.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await scrape

            self.release_slow.set()
            await asyncio.gather(*orchestration._late_collections)

        self.assertEqual(self.cached, [["fast", "slow"]])


class LiveResponseDeadlineSettingTests(unittest.TestCase):
    def test_unset_or_blank_disables_it(self):
        self.assertIsNone(AppSettings.validate_live_scrape_response_deadline(None))
        self.assertIsNone(AppSettings.validate_live_scrape_response_deadline(" "))

    def test_accepts_positive_seconds(self):
        self.assertEqual(AppSettings.validate_live_scrape_response_deadline("4.5"), 4.5)

    def test_rejects_zero_and_negative(self):
        for value in ("0", -1, "nan"):
            with self.assertRaises(ValueError):
                AppSettings.validate_live_scrape_response_deadline(value)
