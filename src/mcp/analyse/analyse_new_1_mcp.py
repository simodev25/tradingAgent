from datetime import datetime, timedelta
import re

def _parse_pair(text: str) -> str:
    """
    Accepte 'eurusd', 'eur/usd', 'eur usd'… -> normalise en 'EUR/USD'
    """
    t = text.lower()
    m = re.search(r'(eur)\s*[/\s_-]?\s*(usd)', t, re.I)
    if m: return 'EUR/USD'
    # Ajoutez d’autres paires ici si besoin
    raise ValueError("Impossible de détecter la paire dans l'entrée.")

def _parse_date(text: str) -> datetime | None:
    """
    Supporte dd/mm/yyyy, dd-mm-yyyy, dd.mm.yyyy, 'today', 'yesterday'.
    Si rien: retourne None (à traiter côté appelant).
    """
    t = text.strip().lower()
    if any(w in t for w in ["today", "aujourd", "ajd"]):
        return datetime.utcnow().date()
    if "yesterday" in t or "hier" in t:
        return (datetime.utcnow() - timedelta(days=1)).date()

    # dd/mm/yyyy (priorité format FR)
    m = re.search(r'(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})', t)
    if m:
        d, mth, y = map(int, m.groups())
        if y < 100: y += 2000
        return datetime(year=y, month=mth, day=d).date()
    return None

def build_search_plan(user_query: str, lang: str = "fr"):
    """
    Transforme une entrée du type 'eurusd 29/10/2026' en:
      - fenêtre de dates (J, J±1)
      - liste courte de requêtes prêtes à lancer
      - garde-fous simples (cap par domaine, etc.)
    """
    # 1) Normalisation
    uq = user_query.strip()
    pair = _parse_pair(uq)
    date_obj = _parse_date(uq)  # None si absente

    # 2) Fenêtre de temps
    if date_obj:
        start = date_obj
        end   = date_obj + timedelta(days=1)  # [J ; J+1)
        date_str_iso = date_obj.strftime("%Y-%m-%d")
        date_str_human = date_obj.strftime("%d %b %Y")
    else:
        # par défaut: aujourd'hui ±1 jour pour couvrir fuseaux/latences
        today = datetime.utcnow().date()
        start = today - timedelta(days=1)
        end   = today + timedelta(days=1)
        date_str_iso = ""
        date_str_human = ""

    # 3) Variantes utiles pour les moteurs (avec/sans slash, mot-clé)
    pair_tokens = {
        "compact": pair.replace("/", ""),
        "spaced": pair.replace("/", " "),
        "slash": pair,
        "words_fr": "euro dollar",
        "words_en": "euro dollar"
    }

    # 4) Génération d’un petit pack de requêtes (8 max)
    base_terms = [
        "{p_slash} forex news",
        "{p_compact} news",
        "{p_spaced} central bank decision",
        "{p_spaced} CPI PMI GDP",
        "{p_spaced} rate hike cut guidance",
        "{p_spaced} FX news",
        "{p_spaced} site:reuters.com",
        "{p_spaced} site:bloomberg.com",
    ]

    # Date hint facultatif dans la query elle-même (aide le ranking)
    date_hint = f" {date_str_human}" if date_str_human else ""

    def fmt(s: str):
        return s.format(
            p_slash=pair_tokens["slash"],
            p_compact=pair_tokens["compact"],
            p_spaced=pair_tokens["spaced"],
        ).strip() + date_hint

    queries = [fmt(s) for s in base_terms]

    # 5) Plan final
    plan = {
        "pair": pair,
        "time_window": {
            "from": start.strftime("%Y-%m-%d"),
            "to":   end.strftime("%Y-%m-%d")
        },
        "language": lang,
        "queries": queries[:8],            # cap dur
        "max_parallel": 4,                 # pour éviter les timeouts
        "per_domain_cap": 2,               # Reuters/Bbg ne monopolisent pas
        "ignore_patterns": [
            "site:ft.com",                 # paywall
            "site:tradingview.com/news"   # index vide/malsain
        ]
    }
    return plan

# --- Exemple d'usage ---
if __name__ == "__main__":
    print(build_search_plan("eurusd 29/10/2026"))
    # print(build_search_plan("eurusd today"))
