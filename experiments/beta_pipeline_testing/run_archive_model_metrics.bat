@echo off
setlocal
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%archive_model_metrics.py" --output-root "D:\GitRepos\runs_pipeline" --clean %*
endlocal
