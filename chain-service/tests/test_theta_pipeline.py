"""theta must survive the whole path to the deliverable, not just the hit
dict: qualifying_puts -> screen_results.json -> rank() -> screen_ranked.json
-> the sidecar's _load_ranked. rank_shortlist copies hits with {**hit, ...}
and the API serves the file verbatim, so nothing should drop it — this
test is what keeps that true if either ever starts picking fields."""

import json
from unittest.mock import patch

from conftest import make_chain, make_put, spot_of

import api
from manifest import new_manifest, write_manifest
from put_filters import qualifying_puts
from rank_shortlist import rank

RUN_ID = "2023-11-14T00:00:00+00:00"


def test_theta_reaches_screen_ranked_json_and_the_api_loader(tmp_path):
    hits = []
    for symbol, spot in (("AAA", 100.0), ("BBB", 100.0)):
        chain = make_chain([make_put()], spot=spot, symbol=symbol)
        hits += qualifying_puts(chain, spot_of(chain))
    assert len(hits) == 2 and all(h["theta"] < 0 for h in hits)

    results, manifest, out = (tmp_path / n for n in
                              ("screen_results.json", "run_manifest.json", "screen_ranked.json"))
    results.write_text(json.dumps(hits))
    write_manifest(
        new_manifest(RUN_ID, "abc", 503)
        | {"tickers_screened": 503, "tickers_errored": 0, "qualifying_contracts": 2},
        manifest,
    )

    with patch("rank_shortlist.get_next_earnings_date", return_value="2033-12-31"):
        ranked, _ = rank(results, out, manifest, api_key="test-key", sleep=0)

    on_disk = json.loads(out.read_text())
    served = api._load_ranked(tmp_path)  # what GET /runs/{id} puts in body.ranked
    for rows in (ranked, on_disk, served):
        assert len(rows) == 2
        by_symbol = {r["underlyingSymbol"]: r for r in rows}
        for original in hits:
            assert by_symbol[original["underlyingSymbol"]]["theta"] == original["theta"]
