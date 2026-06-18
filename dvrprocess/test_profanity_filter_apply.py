#!/usr/bin/env python3
import importlib.util
import pathlib
import threading
import time
import unittest


def _load_profanity_filter_apply():
    path = pathlib.Path(__file__).with_name('profanity-filter-apply.py')
    spec = importlib.util.spec_from_file_location('profanity_filter_apply', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.compute_filter_hash = lambda: 'current-hash'
    return module


profanity_filter_apply = _load_profanity_filter_apply()


class MediaItem:
    def __init__(self, name: str, tags: dict[str, str]):
        self.name = name
        self.tags = tags
        self.host_file_path = name


class ProfanityFilterApplyTest(unittest.TestCase):

    def setUp(self) -> None:
        self.module = profanity_filter_apply
        self.constants = self.module.constants
        self.selector = getattr(self.module, '__profanity_filter_selector')
        self.current_tags = {
            self.constants.K_FILTER_HASH: 'current-hash',
            self.constants.K_FILTER_VERSION: '999',
        }

    def _unfiltered_item(self, name='unfiltered'):
        return MediaItem(name, {})

    def _new_version_item(self, name='new-version'):
        tags = dict(self.current_tags)
        tags[self.constants.K_FILTER_VERSION] = '0'
        return MediaItem(name, tags)

    def _config_change_item(self, name='config-change'):
        tags = dict(self.current_tags)
        tags[self.constants.K_FILTER_HASH] = 'old-hash'
        return MediaItem(name, tags)

    def test_selector_yields_lower_priority_item_before_scan_completes(self):
        release_scan = threading.Event()

        def blocked_scan():
            yield self._config_change_item('config-first')
            release_scan.wait(timeout=5)
            yield self._unfiltered_item('unfiltered-second')

        selector = self.selector(
            blocked_scan(), set(self.module.ProfanityFilterSelector))
        try:
            started = time.monotonic()
            first = next(selector)
            elapsed = time.monotonic() - started

            self.assertEqual('config-first', first.name)
            self.assertLess(elapsed, 1)
        finally:
            release_scan.set()
            selector.close()

    def test_selector_uses_priority_order_for_available_items(self):
        def scanned_items():
            yield self._config_change_item()
            yield self._new_version_item()
            yield self._unfiltered_item()

        selected = list(self.selector(
            scanned_items(), set(self.module.ProfanityFilterSelector)))

        self.assertEqual(
            ['unfiltered', 'new-version', 'config-change'],
            [item.name for item in selected])

    def test_selector_honors_requested_selectors(self):
        def scanned_items():
            yield self._unfiltered_item()
            yield self._new_version_item()
            yield self._config_change_item()

        selected = list(self.selector(
            scanned_items(), {self.module.ProfanityFilterSelector.config_change}))

        self.assertEqual(['config-change'], [item.name for item in selected])


if __name__ == '__main__':
    unittest.main()
