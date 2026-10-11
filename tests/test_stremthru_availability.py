import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from comet.debrid import stremthru
from comet.debrid.exceptions import DebridAuthError, DebridLinkGenerationError
from comet.debrid.stremthru import (
    StremThru,
    _prepare_cached_torrents,
)


class _ResponseContext:
    def __init__(self, payload=None, *, status=200, error=None, text="raw"):
        self.payload = payload
        self.status = status
        self.error = error
        self.raw_text = text
        self.exited = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.exited = True

    async def json(self, **kwargs):
        del kwargs
        if self.error is not None:
            raise self.error
        return self.payload

    async def text(self):
        return self.raw_text


class _Session:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        del args, kwargs
        return self.response

    def post(self, *args, **kwargs):
        del args, kwargs
        return self.response


class StremThruAvailabilityTests(unittest.TestCase):
    def test_malformed_torrents_and_files_are_isolated_once(self):
        responses = [
            None,
            {"data": {"items": "invalid"}},
            {
                "data": {
                    "items": [
                        {"status": "cached", "files": []},
                        {
                            "status": "cached",
                            "hash": "a" * 40,
                            "files": [
                                None,
                                {"name": "Sample.mkv", "index": 0, "size": 10},
                                {
                                    "name": "folder/First.S01E01.mkv",
                                    "index": 1,
                                    "size": 20,
                                },
                                {"name": 42},
                                {"name": "Second.S01E02.MP4", "index": 2, "size": 30},
                            ],
                        },
                        {"status": "downloading", "hash": "b" * 40, "files": []},
                    ]
                }
            },
        ]

        torrents, filenames = _prepare_cached_torrents(
            responses,
            is_offcloud=False,
        )

        self.assertEqual(filenames, ["First.S01E01.mkv", "Second.S01E02.MP4"])
        self.assertEqual([torrent["info_hash"] for torrent in torrents], ["a" * 40])
        self.assertEqual(
            [filename for _, filename in torrents[0]["files"]],
            filenames,
        )


class StremThruResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_store_json_response_closes_after_complete_payload_read(self):
        response = _ResponseContext({"data": {"value": "complete"}})
        client = StremThru(_Session(response), None, None, "realdebrid:token", "")

        payload = await client._post_store_json("/endpoint", {}, "read store")

        self.assertEqual(payload, {"data": {"value": "complete"}})
        self.assertTrue(response.exited)

    async def test_premium_response_closes_on_auth_error(self):
        response = _ResponseContext({"error": "invalid"})
        client = StremThru(_Session(response), None, None, "realdebrid:token", "")

        with self.assertRaises(DebridAuthError):
            await client.check_premium()

        self.assertTrue(response.exited)

    async def test_instant_response_closes_on_json_error(self):
        response = _ResponseContext(error=ValueError("invalid JSON"))
        client = StremThru(_Session(response), None, None, "realdebrid:token", "")

        self.assertIsNone(await client.get_instant(["a" * 40]))
        self.assertTrue(response.exited)

    async def test_magnet_list_response_closes_on_invalid_payload(self):
        response = _ResponseContext({"data": None})
        client = StremThru(_Session(response), None, None, "realdebrid:token", "")

        self.assertEqual(await client.list_magnets(), (None, 0))
        self.assertTrue(response.exited)

    async def test_unexpected_link_error_is_typed_and_visible(self):
        client = StremThru(None, None, None, "realdebrid:token", "")
        with (
            patch.object(
                client,
                "_post_store_json",
                new=AsyncMock(side_effect=RuntimeError("transport failed")),
            ),
            self.assertRaises(DebridLinkGenerationError) as raised,
        ):
            await client.generate_download_link(
                "a" * 40,
                "0",
                "Movie.mkv",
                "Movie",
                None,
                None,
            )

        self.assertEqual(raised.exception.payload["error_type"], "RuntimeError")
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)


class _CountingSession:
    def __init__(self, payload):
        self.payload = payload
        self.urls = []

    def get(self, url, *args, **kwargs):
        del args, kwargs
        self.urls.append(url)
        return _ResponseContext(self.payload)


class StremThruPremiumCheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        stremthru._premium_verified_until.clear()

    async def test_premium_status_is_reused_for_the_same_account(self):
        session = _CountingSession({"data": {"subscription_status": "premium"}})

        await StremThru(session, None, None, "torbox:token", "").check_premium()
        await StremThru(session, None, None, "torbox:token", "").check_premium()
        await StremThru(session, None, None, "torbox:other", "").check_premium()

        self.assertEqual(len(session.urls), 2)

    async def test_failed_premium_check_is_not_reused(self):
        session = _CountingSession({"data": {"subscription_status": "expired"}})

        for _ in range(2):
            with self.assertRaises(DebridAuthError):
                await StremThru(session, None, None, "torbox:token", "").check_premium()

        self.assertEqual(len(session.urls), 2)

    async def test_premium_check_and_availability_lookups_run_together(self):
        client = StremThru(None, None, None, "torbox:token", "")
        both_started = asyncio.Event()
        started = []

        async def premium():
            started.append("premium")
            if len(started) == 2:
                both_started.set()
            await both_started.wait()

        async def instant(chunk):
            del chunk
            started.append("instant")
            if len(started) == 2:
                both_started.set()
            await both_started.wait()
            return None

        with (
            patch.object(client, "check_premium", new=premium),
            patch.object(client, "get_instant", new=instant),
        ):
            await asyncio.wait_for(
                client.get_availability(["a" * 40], {}, {}, {}), timeout=1
            )

        self.assertCountEqual(started, ["premium", "instant"])

    async def test_failed_premium_check_fails_the_availability_call(self):
        client = StremThru(None, None, None, "torbox:token", "")

        with (
            patch.object(
                client,
                "check_premium",
                new=AsyncMock(side_effect=DebridAuthError("torbox", "expired")),
            ),
            patch.object(client, "get_instant", new=AsyncMock(return_value=None)),
            self.assertRaises(DebridAuthError),
        ):
            await client.get_availability(["a" * 40], {}, {}, {})
