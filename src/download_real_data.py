import os
import wfdb

# Directorio de salida
destino_cinc = "data/cinc2013_real"
os.makedirs(destino_cinc, exist_ok=True)

# -------------------------------------------------------------
# Descarga de los 75 registros de CinC Challenge 2013 (Set-a)
# -------------------------------------------------------------
# Cada registro se identifica como 'set-a/a01' ... 'set-a/a75'
registros_cinc = [f"set-a/a{i:02d}" for i in range(1, 76)]

print(f"Iniciando descarga de los {len(registros_cinc)} registros reales de CinC 2013...")

try:
    wfdb.dl_database(
        "challenge-2013",
        dl_dir=destino_cinc,
        records=registros_cinc
    )
    print("\n✅ Los 75 registros de CinC Challenge 2013 se descargaron correctamente.")
except Exception as e:
    print(f"\n⚠️ Error durante la descarga: {e}")