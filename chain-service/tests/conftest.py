import json
import sys
from pathlib import Path

import pytest

CHAIN_SERVICE = Path(__file__).resolve().parent.parent
FIXTURES = CHAIN_SERVICE.parent / "fixtures"

sys.path.insert(0, str(CHAIN_SERVICE))


def load_chain(name):
    with open(FIXTURES / f"{name}-chain.json") as f:
        return json.load(f)


def spot_of(chain):
    return chain["optionChain"]["result"][0]["quote"]["regularMarketPrice"]


@pytest.fixture
def aapl():
    return load_chain("AAPL")


@pytest.fixture
def nvr():
    """NVR returned a result object with zero expirationDates and no options."""
    return load_chain("NVR")


@pytest.fixture
def brk_b():
    """Saved as BRK.B-chain.json, but Yahoo reports underlyingSymbol as BRK-B."""
    return load_chain("BRK.B")


def make_chain(puts, spot=100.0, as_of=1_700_000_000, expiration=None, symbol="TEST"):
    """Build a minimal chain shaped like Yahoo's response.

    Defaults put expiration 14 days after as_of, matching the DTE target the
    pipeline aims for, so delta lands in a realistic range.
    """
    if expiration is None:
        expiration = as_of + 14 * 86400
    return {
        "optionChain": {
            "result": [
                {
                    "underlyingSymbol": symbol,
                    "quote": {"regularMarketPrice": spot, "regularMarketTime": as_of},
                    "expirationDates": [expiration],
                    "options": [{"expirationDate": expiration, "puts": puts}],
                }
            ]
        }
    }


def make_put(**overrides):
    """A put that passes every filter, so a test can break exactly one thing.

    spot defaults to 100 in make_chain: strike 85 is 15% OTM, bid 1.50 gives
    a return of 1.50/83.50 = 1.8%, and 50k/(85*100) = 5 contracts, so the
    liquidity floors are oi>=50 and volume>=10.
    """
    put = {
        "contractSymbol": "TEST240101P00085000",
        "strike": 85.0,
        "bid": 1.50,
        "impliedVolatility": 0.35,
        "openInterest": 500,
        "volume": 100,
    }
    put.update(overrides)
    return put
