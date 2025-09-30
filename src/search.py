import pandas as pd
import requests
import time
import re

# Cargar archivo CSV
df = pd.read_csv("games.csv")

# Diccionario donde se almacenarán los años encontrados
release_years = {}

# Total de juegos para mostrar el progreso
total = len(df)

# Búsqueda básica usando Wikipedia API
def fetch_release_year(name, platform):
    q = f"{name.replace('_', ' ')} {platform}"
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": q,
        "format": "json",
        "srlimit": 1
    }
    try:
        r = requests.get(url, params=params, timeout=10).json()
        if r['query']['search']:
            page_title = r['query']['search'][0]['title']
            detail = requests.get(url, params={
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "titles": page_title,
                "format": "json"
            }, timeout=10).json()

            content = next(iter(detail['query']['pages'].values()))['revisions'][0]['*']
            match = re.search(r'released?.*?(\d{4})', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
    except Exception as e:
        print(f"⚠️ Error con '{name} ({platform})': {e}")
    return None

# Bucle de búsqueda con avance y delay
for i, (name, platform) in enumerate(zip(df['name'], df['platform']), start=1):
    key = (name, platform)
    print(f"[{i}/{total}] Buscando: {name} ({platform})")
    
    if key not in release_years:
        release_years[key] = fetch_release_year(name, platform) or 'unknown'
        time.sleep(1)  # Pausa para no ser bloqueado

# Añadir columna al DataFrame
df['release_year'] = df.apply(lambda r: release_years.get((r['name'], r['platform']), 'unknown'), axis=1)

# Reordenar columnas y guardar archivo
cols = [c for c in df.columns if c != 'release_year'] + ['release_year']
df = df[cols]
df.to_csv("games_with_year.csv", index=False)
print("✅ Archivo guardado como 'games_with_year.csv'")
