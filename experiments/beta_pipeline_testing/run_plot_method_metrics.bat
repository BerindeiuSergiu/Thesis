@echo off
setlocal
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%plot_method_metrics.py" %*
endlocal
