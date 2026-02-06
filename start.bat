@echo off
setlocal enabledelayedexpansion

:: ============================================
:: DeepECG - Script de demarrage
:: ============================================

title DeepECG Launcher

echo.
echo  ========================================
echo   DeepECG - Medical AI Analysis Platform
echo  ========================================
echo.

:: Couleurs pour les messages
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "CYAN=[96m"
set "RESET=[0m"

:: Verification du repertoire
cd /d "%~dp0"
echo %CYAN%[INFO]%RESET% Repertoire de travail: %CD%
echo.

:: Menu de selection du mode
echo Choisissez le mode de lancement:
echo.
echo   1. Docker Compose (recommande - tout conteneurise)
echo   2. Mode developpement (backend + frontend locaux)
echo   3. Backend uniquement (Python local)
echo   4. Frontend uniquement (npm local)
echo   5. AI Engine Docker uniquement (GPU)
echo   6. Dev complet (AI Engine + Backend + Frontend)
echo   7. Quitter
echo.
set /p MODE="Votre choix [1-7]: "

if "%MODE%"=="1" goto :docker_mode
if "%MODE%"=="2" goto :dev_mode
if "%MODE%"=="3" goto :backend_only
if "%MODE%"=="4" goto :frontend_only
if "%MODE%"=="5" goto :ai_engine_only
if "%MODE%"=="6" goto :full_dev_mode
if "%MODE%"=="7" goto :exit
echo %RED%[ERREUR]%RESET% Choix invalide
goto :exit

:: ============================================
:: Mode Docker Compose
:: ============================================
:docker_mode
echo.
echo %CYAN%[INFO]%RESET% Lancement en mode Docker Compose...
echo.

:: Verifier Docker
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Docker n'est pas installe ou pas dans le PATH
    echo Veuillez installer Docker Desktop: https://www.docker.com/products/docker-desktop
    goto :exit
)

:: Verifier que Docker est en cours d'execution
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Docker n'est pas demarre
    echo Veuillez demarrer Docker Desktop
    goto :exit
)

echo %GREEN%[OK]%RESET% Docker detecte
echo.

:: Construire et lancer les containers
echo %CYAN%[INFO]%RESET% Construction des images Docker...
docker-compose build

if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Echec de la construction des images
    goto :exit
)

echo.
echo %CYAN%[INFO]%RESET% Demarrage des services...
docker-compose up -d

if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Echec du demarrage des services
    goto :exit
)

echo.
echo %GREEN%========================================%RESET%
echo %GREEN%  DeepECG demarre avec succes!%RESET%
echo %GREEN%========================================%RESET%
echo.
echo   Frontend:   http://localhost:5173
echo   Backend:    http://localhost:8000
echo   API Docs:   http://localhost:8000/docs
echo   AI Engine:  http://localhost:8001
echo.
echo   Pour arreter: docker-compose down
echo   Pour les logs: docker-compose logs -f
echo.
goto :exit

:: ============================================
:: Mode Developpement (Backend + Frontend)
:: ============================================
:dev_mode
echo.
echo %CYAN%[INFO]%RESET% Lancement en mode developpement...
echo.

:: Verifier Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Python n'est pas installe ou pas dans le PATH
    goto :exit
)
echo %GREEN%[OK]%RESET% Python detecte

:: Verifier Node.js
where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Node.js/npm n'est pas installe ou pas dans le PATH
    goto :exit
)
echo %GREEN%[OK]%RESET% Node.js/npm detecte
echo.

:: Installer les dependances backend si necessaire
if not exist "backend\venv" (
    echo %CYAN%[INFO]%RESET% Creation de l'environnement virtuel Python...
    python -m venv backend\venv
)

echo %CYAN%[INFO]%RESET% Installation des dependances backend...
call backend\venv\Scripts\activate.bat
pip install -r backend\requirements.txt -q

:: Installer les dependances frontend si necessaire
if not exist "frontend\node_modules" (
    echo %CYAN%[INFO]%RESET% Installation des dependances frontend...
    cd frontend
    call npm install
    cd ..
)

echo.
echo %CYAN%[INFO]%RESET% Demarrage des services...
echo.

:: Creer le repertoire temporaire
if not exist "%TEMP%\deepecg" mkdir "%TEMP%\deepecg"

:: Lancer le backend dans une nouvelle fenetre
start "DeepECG Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && set DEBUG=true && set AI_ENGINE_URL=http://localhost:8001 && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

:: Attendre que le backend demarre
echo %YELLOW%[WAIT]%RESET% Attente du demarrage du backend...
timeout /t 3 /nobreak >nul

:: Lancer le frontend dans une nouvelle fenetre
start "DeepECG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo %GREEN%========================================%RESET%
echo %GREEN%  DeepECG demarre avec succes!%RESET%
echo %GREEN%========================================%RESET%
echo.
echo   Frontend:   http://localhost:5173
echo   Backend:    http://localhost:8000
echo   API Docs:   http://localhost:8000/docs
echo.
echo   %YELLOW%Note: L'AI Engine doit etre demarre separement%RESET%
echo.
echo   Fermez les fenetres de terminal pour arreter
echo.
goto :exit

:: ============================================
:: Backend uniquement
:: ============================================
:backend_only
echo.
echo %CYAN%[INFO]%RESET% Lancement du backend uniquement...
echo.

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Python n'est pas installe
    goto :exit
)

if not exist "backend\venv" (
    echo %CYAN%[INFO]%RESET% Creation de l'environnement virtuel...
    python -m venv backend\venv
)

echo %CYAN%[INFO]%RESET% Installation des dependances...
call backend\venv\Scripts\activate.bat
pip install -r backend\requirements.txt -q

if not exist "%TEMP%\deepecg" mkdir "%TEMP%\deepecg"

echo.
echo %GREEN%[OK]%RESET% Demarrage du backend sur http://localhost:8000
echo %GREEN%[OK]%RESET% Documentation API: http://localhost:8000/docs
echo.

cd backend
set DEBUG=true
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
goto :exit

:: ============================================
:: Frontend uniquement
:: ============================================
:frontend_only
echo.
echo %CYAN%[INFO]%RESET% Lancement du frontend uniquement...
echo.

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Node.js/npm n'est pas installe
    goto :exit
)

if not exist "frontend\node_modules" (
    echo %CYAN%[INFO]%RESET% Installation des dependances...
    cd frontend
    call npm install
    cd ..
)

echo.
echo %GREEN%[OK]%RESET% Demarrage du frontend sur http://localhost:5173
echo.

cd frontend
npm run dev
goto :exit

:: ============================================
:: AI Engine Docker uniquement
:: ============================================
:ai_engine_only
echo.
echo %CYAN%[INFO]%RESET% Lancement du AI Engine Docker...
echo.

:: Verifier Docker
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Docker n'est pas installe
    goto :exit
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Docker n'est pas demarre
    goto :exit
)

echo %GREEN%[OK]%RESET% Docker detecte
echo.

:: Verifier si PowerShell est disponible
where powershell >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% PowerShell n'est pas disponible
    goto :exit
)

echo %CYAN%[INFO]%RESET% Demarrage du AI Engine avec GPU...
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0start-ai-engine.ps1" -Detached

echo.
echo %GREEN%[OK]%RESET% AI Engine demarre sur http://localhost:8001
echo %GREEN%[OK]%RESET% Health check: http://localhost:8001/health
echo.
goto :exit

:: ============================================
:: Mode Dev Complet (AI + Backend + Frontend)
:: ============================================
:full_dev_mode
echo.
echo %CYAN%[INFO]%RESET% Lancement du mode developpement complet...
echo.

:: Verifier tous les prerequis
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Docker n'est pas installe
    goto :exit
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Python n'est pas installe
    goto :exit
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Node.js/npm n'est pas installe
    goto :exit
)

docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo %RED%[ERREUR]%RESET% Docker n'est pas demarre
    goto :exit
)

echo %GREEN%[OK]%RESET% Tous les prerequis detectes
echo.

:: 1. Demarrer AI Engine
echo %CYAN%[INFO]%RESET% Demarrage du AI Engine Docker...
powershell -ExecutionPolicy Bypass -File "%~dp0start-ai-engine.ps1" -Detached

echo %YELLOW%[WAIT]%RESET% Attente du demarrage du AI Engine...
timeout /t 10 /nobreak >nul

:: 2. Preparer et demarrer Backend
if not exist "backend\venv" (
    echo %CYAN%[INFO]%RESET% Creation de l'environnement virtuel Python...
    python -m venv backend\venv
)

echo %CYAN%[INFO]%RESET% Installation des dependances backend...
call backend\venv\Scripts\activate.bat
pip install -r backend\requirements.txt -q

if not exist "%TEMP%\deepecg" mkdir "%TEMP%\deepecg"

start "DeepECG Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && set DEBUG=true && set AI_ENGINE_URL=http://localhost:8001 && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo %YELLOW%[WAIT]%RESET% Attente du demarrage du backend...
timeout /t 5 /nobreak >nul

:: 3. Preparer et demarrer Frontend
if not exist "frontend\node_modules" (
    echo %CYAN%[INFO]%RESET% Installation des dependances frontend...
    cd frontend
    call npm install
    cd ..
)

start "DeepECG Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo %GREEN%========================================%RESET%
echo %GREEN%  DeepECG - Mode Complet Demarre!%RESET%
echo %GREEN%========================================%RESET%
echo.
echo   AI Engine:  http://localhost:8001
echo   Backend:    http://localhost:8000
echo   Frontend:   http://localhost:5173
echo   API Docs:   http://localhost:8000/docs
echo.
echo   Pour arreter:
echo   - Fermez les fenetres de terminal
echo   - docker stop deepecg-ai-engine
echo.
goto :exit

:: ============================================
:: Sortie
:: ============================================
:exit
echo.
pause
endlocal
