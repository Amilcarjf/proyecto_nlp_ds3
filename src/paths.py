from pathlib import Path


# ==========================================
# Raíz del proyecto
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================
# Datos
# ==========================================

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"


# ==========================================
# Salidas
# ==========================================

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_FIGURES = OUTPUTS_DIR / "figures"
OUTPUTS_TABLES = OUTPUTS_DIR / "tables"
OUTPUTS_METRICS = OUTPUTS_DIR / "metrics"
OUTPUTS_MODELS = OUTPUTS_DIR / "models"
OUTPUTS_LOGS = OUTPUTS_DIR / "logs"


# ==========================================
# Informes
# ==========================================

REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_output_directories():
    """
    Crea únicamente los directorios de salida autorizados.
    No crea ni modifica la raíz del proyecto.
    """

    directories = [
        DATA_INTERIM,
        DATA_PROCESSED,
        OUTPUTS_FIGURES,
        OUTPUTS_TABLES,
        OUTPUTS_METRICS,
        OUTPUTS_MODELS,
        OUTPUTS_LOGS,
        REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)