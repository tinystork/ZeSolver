<#
  Installer toutes les dépendances du projet (Windows, PowerShell)
  - Crée un venv .venv (Python 3.13 si disponible)
  - Installe requirements.txt
  - Installe Twirl SANS dépendances (pour ignorer son pin astropy<6)
  - Vérifie l’installation

  Usage:
    .\install.ps1
#>

$ErrorActionPreference = "Stop"

# --- Paramètres ---
$VenvPath = ".venv"
$ReqFile  = "requirements.txt"
$TwirlVer = "0.4.2"

Write-Host "=== Étape 1/5 : Détection Python ==="
# Privilégie le launcher Windows 'py' avec 3.13 si présent
$pythonCmd = $null
try {
    $pyList = & py -0p 2>$null
    if ($pyList -match "3.13") {
        $pythonCmd = "py -3.13"
    } elseif ($pyList) {
        $pythonCmd = "py"
    }
} catch { }

if (-not $pythonCmd) {
    # fallback
    $pythonCmd = "python"
}
Write-Host "Python choisi : $pythonCmd"

Write-Host "=== Étape 2/5 : Création/activation venv ==="
if (-not (Test-Path $VenvPath)) {
    & $pythonCmd -m venv $VenvPath
    Write-Host "Venv créé: $VenvPath"
}
# Active le venv pour la session courante
$activate = Join-Path $VenvPath "Scripts\Activate.ps1"
. $activate
Write-Host "Venv activé: $VenvPath"

Write-Host "=== Étape 3/5 : Mise à jour de l’outillage ==="
python -m pip install -U pip setuptools wheel

Write-Host "=== Étape 4/5 : Installation des dépendances ==="
if (-not (Test-Path $ReqFile)) {
    Write-Error "Fichier $ReqFile introuvable !"
}
# Installe TOUTES les deps normales
python -m pip install -r $ReqFile
# Installe Twirl SANS deps (pour garder Astropy 7)
python -m pip install --no-deps "twirl==$TwirlVer"

Write-Host "=== Étape 5/5 : Vérifications ==="
$code = @"
import importlib.metadata as m
import sys, astropy
import twirl
print("OK Python:", sys.version.split()[0])
print("OK Astropy:", astropy.__version__)
print("OK Twirl:", m.version("twirl"))
print("Twirl module path:", twirl.__file__)
"@
python - <<PY
$code
PY

Write-Host "`n✅ Installation terminée."
Write-Host "Pour utiliser l'environnement plus tard :"
Write-Host "  .\.venv\Scripts\Activate.ps1"
