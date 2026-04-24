import unittest
from unittest.mock import patch

import market_context


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class MarketContextTests(unittest.TestCase):
    def test_bybit_linear_btcusdt_open_interest_is_already_btc(self):
        payload = {
            "result": {
                "list": [
                    {"openInterest": "71234.5", "timestamp": "1777050000000"},
                ]
            }
        }
        ctx = {"open_interest": 65000.0, "mark_price": 78000.0}

        with patch.object(market_context._http, "get", return_value=_FakeResponse(payload)):
            market_context._fetch_bybit_oi(ctx)

        self.assertEqual(ctx["bybit_oi"], 71234.5)
        self.assertEqual(ctx["combined_oi"], 136234.5)


if __name__ == "__main__":
    unittest.main()
