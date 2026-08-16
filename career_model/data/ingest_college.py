"""College production, strength-of-schedule adjusted.  Prior covariates only.

These feed `x_i` (§4.3) and nothing else.  College BPM is legitimate here in a
way NBA BPM never is: it summarises a period the filter does not run over, so
it cannot double-count shrinkage or collapse the state (§1.2, §1.3).

Source: barttorvik.com's per-season advanced CSV, cached under
`data_raw/college/`.  Torvik's `bpm`/`obpm`/`dbpm`/`adjoe` are already
opponent- and tempo-adjusted; on top of that we add a conference-strength
z-score and an age-adjusted production score, because "adjusted" metrics still
leave a readable conference residual and a 19-year-old sophomore posting a
given line is not the same prospect as a 23-year-old senior posting it.

Coverage starts in 2008.  Everyone else -- internationals, G-League Ignite,
prep-to-pro, and anyone whose last college season predates the coverage window
-- gets NaN, which `build_panel` turns into an explicit missingness indicator
rather than an imputed zero (§4.3).
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from ..config import RAW_DIR

COLLEGE_DIR = RAW_DIR / "college"
BT_FIRST_YEAR = 2008
BT_LAST_YEAR = 2026

# The endpoint returns a headerless CSV; this 67-field order is stable.
BT_COLUMNS = [
    "player_name", "team", "conf", "gp", "min_pct", "ortg", "usg", "efg", "ts_pct",
    "oreb_pct", "dreb_pct", "ast_pct", "tov_pct", "ftm", "fta", "ft_pct",
    "two_pm", "two_pa", "two_pct", "tpm", "tpa", "tp_pct", "blk_pct", "stl_pct",
    "ftr", "yr", "ht", "num", "porpag", "adjoe", "pfr", "year", "pid", "hometown",
    "rec_rank", "ast_tov", "rim_made", "rim_att", "mid_made", "mid_att",
    "rim_pct", "mid_pct", "dunks_made", "dunks_att", "dunk_pct", "pick",
    "drtg", "adrtg", "dporpag", "stops", "bpm", "obpm", "dbpm", "gbpm", "mpg",
    "ogbpm", "dgbpm", "oreb_pg", "dreb_pg", "treb_pg", "ast_pg", "stl_pg",
    "blk_pg", "pts_pg", "role", "threshold", "birthdate",
]
_TEXT = {"player_name", "team", "conf", "yr", "ht", "hometown", "role", "birthdate"}
_NUMERIC = [c for c in BT_COLUMNS if c not in _TEXT]


def _log(msg: str) -> None:
    print(f"[college] {msg}", flush=True)


def normalize_name(name: str) -> str:
    """Accent- and punctuation-insensitive key.  'Nikola Jokić' -> 'nikolajokic'."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    for suffix in (" jr.", " jr", " sr.", " sr", " iii", " ii", " iv"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return "".join(ch for ch in s if ch.isalnum())


def load_raw(first_year: int = BT_FIRST_YEAR, last_year: int = BT_LAST_YEAR) -> pd.DataFrame:
    frames = []
    for year in range(first_year, last_year + 1):
        path = COLLEGE_DIR / f"barttorvik_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, header=None, names=BT_COLUMNS, low_memory=False)
        for c in _NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["season_year"] = year
        frames.append(df)
    if not frames:
        _log(f"no cached college seasons under {COLLEGE_DIR}")
        return pd.DataFrame(columns=[*BT_COLUMNS, "season_year"])
    out = pd.concat(frames, ignore_index=True)
    _log(f"{len(out)} college player-seasons, {out['season_year'].min()}-{out['season_year'].max()}")
    return out


def add_adjustments(df: pd.DataFrame, min_minutes_pct: float = 40.0) -> pd.DataFrame:
    """Conference strength, SOS-corrected BPM, and age-adjusted production."""
    df = df.copy()

    rot = df[df["min_pct"] >= min_minutes_pct]
    conf = rot.groupby(["season_year", "conf"])["bpm"].mean().rename("conf_bpm_mean").reset_index()
    conf["conf_strength"] = conf.groupby("season_year")["conf_bpm_mean"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
    df = df.merge(conf[["season_year", "conf", "conf_strength"]],
                  on=["season_year", "conf"], how="left")

    # One sd of conference strength is worth roughly one BPM point -- fit the
    # slope rather than asserting it.  We *add back* half the competition level
    # faced; subtracting it outright would double-count Torvik's own adjustment.
    fit = df.dropna(subset=["bpm", "conf_strength"])
    slope = float(np.polyfit(fit["conf_strength"], fit["bpm"], 1)[0]) if len(fit) > 500 else 0.0
    df["bpm_sos"] = df["bpm"] + 0.5 * slope * df["conf_strength"]

    # Age at the end of the college season, from Torvik's birthdate field.
    bd = pd.to_datetime(df["birthdate"], errors="coerce")
    ref = pd.to_datetime(df["season_year"].astype(str) + "-04-01", errors="coerce")
    df["age_final_season"] = (ref - bd).dt.days / 365.25

    # Production relative to what a player that age typically posts.  Residual
    # from a quadratic in age, z-scored within season.
    ok = df["age_final_season"].notna() & df["bpm_sos"].notna() & (df["min_pct"] >= min_minutes_pct)
    df["age_adj_production"] = np.nan
    if ok.sum() > 500:
        a = df.loc[ok, "age_final_season"].to_numpy()
        b = df.loc[ok, "bpm_sos"].to_numpy()
        coef = np.polyfit(a, b, 2)
        expected = np.polyval(coef, df["age_final_season"].to_numpy())
        resid = df["bpm_sos"].to_numpy() - expected
        df["age_adj_production"] = resid
    return df


def build_player_table() -> pd.DataFrame:
    """One row per college player: his final season, plus career aggregates.

    The *final* college season is the right summary because it is the state the
    player enters the NBA in; career-best BPM comes along as a separate column
    for the same reason a scout looks at both.
    """
    raw = load_raw()
    if raw.empty:
        return pd.DataFrame(columns=["name_key"])
    adj = add_adjustments(raw)
    adj["name_key"] = adj["player_name"].map(normalize_name)
    adj = adj[adj["name_key"] != ""]

    adj = adj.sort_values(["name_key", "season_year"])
    last = adj.groupby("name_key").tail(1).set_index("name_key")
    agg = adj.groupby("name_key").agg(
        college_seasons=("season_year", "nunique"),
        college_gp_total=("gp", "sum"),
        college_bpm_best=("bpm_sos", "max"),
    )

    out = pd.DataFrame({
        "college_bpm_sos": last["bpm_sos"],
        "college_obpm": last["obpm"],
        "college_dbpm": last["dbpm"],
        "college_stl_pct": last["stl_pct"],
        "college_blk_pct": last["blk_pct"],
        "college_ft_pct": last["ft_pct"],
        "college_tp_pct": last["tp_pct"],
        "college_ts_pct": last["ts_pct"],
        "college_usg": last["usg"],
        "college_ast_pct": last["ast_pct"],
        "college_oreb_pct": last["oreb_pct"],
        "college_dreb_pct": last["dreb_pct"],
        "college_min_pct": last["min_pct"],
        "college_conf_strength": last["conf_strength"],
        "college_age_adj_production": last["age_adj_production"],
        "college_age_final_season": last["age_final_season"],
        "college_last_year": last["season_year"],
        "college_birthdate": pd.to_datetime(last["birthdate"], errors="coerce"),
    }).join(agg)

    _log(f"{len(out)} distinct college players")
    return out.reset_index()


if __name__ == "__main__":
    t = build_player_table()
    print(t.head().to_string())
