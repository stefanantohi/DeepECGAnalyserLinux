@echo off
:: ============================================
:: DeepECG - Script d'arret
:: ============================================

title DeepECG - Arret

echo.
echo  ========================================
echo   DeepECG - Arret des services
echo  ========================================
echo.

cd /d "%~dp0"

:: Verifier Docker
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERREUR] Docker n'est pas installe
    goto :exit
)

echo [INFO] Arret des containers Docker...
docker-compose down

if %errorlevel% equ 0 (
    echo.
    echo [OK] Tous les services ont ete arretes
) else (
    echo.
    echo [INFO] Aucun container en cours d'execution
)

echo.

:exit
pause
