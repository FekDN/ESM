@echo off
chcp 65001 >nul
setlocal

rem Usage:
rem   run_esm_viewer.bat esm_analysis\runs\my_run\visualization\manifest.json
rem With no argument the default named run is used.
set "view_manifest=%~1"
if "%view_manifest%"=="" set "view_manifest=%CD%\esm_analysis\runs\default\visualization\manifest.json"
if not exist "%view_manifest%" (
    echo Visualisation manifest was not found:
    echo   %view_manifest%
    echo.
    echo First run esm_master_analytics.py or choose an existing manifest.
    echo Legacy visual data, if present, can be opened with:
    echo   run_esm_viewer.bat esm_visualization_data\manifest.json
    exit /b 1
)

echo Starting ESM viewer at http://localhost:8501 ...
start "" http://localhost:8501
wsl -d Ubuntu -e bash -c "cd \"$(wslpath '%CD%')\" && source ~/miniconda3/etc/profile.d/conda.sh && conda activate esm_env && exec streamlit run view_esm_alphabet.py --server.address 0.0.0.0 --server.port 8501 -- --manifest \"$(wslpath '%view_manifest%')\""

if errorlevel 1 (
    echo.
    echo Viewer stopped or could not start. If port 8501 is occupied, close the old Streamlit process.
    exit /b 1
)
