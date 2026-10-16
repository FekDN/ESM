@echo off
chcp 65001 >nul
setlocal
echo Opening ESM control centre in WSL distribution "Ubuntu" ...
echo The browser will open at http://localhost:8503

start "ESM control centre" http://localhost:8503
wsl -d Ubuntu -e bash -c "cd \"$(wslpath '%CD%')\" && source ~/miniconda3/etc/profile.d/conda.sh && conda activate esm_env && exec streamlit run esm_control_center.py --server.address 0.0.0.0 --server.port 8503"

if errorlevel 1 (
  echo.
  echo Streamlit stopped with an error. Check that Ubuntu and the esm_env conda environment exist.
  pause
)
endlocal
