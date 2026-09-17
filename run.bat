@echo off
REM One-command launcher: always uses the project venv (which has torch, netCDF4,
REM segmentation-models-pytorch, etc.), never the conda base python.
REM Usage from the project root:
REM   run.bat --fresh --ensemble        (clean rerun of all experiments end to end)
REM   run.bat                            (resume/continue; skips finished experiments)
cd /d "%~dp0"
REM Expandable CUDA segments avoid the memory fragmentation that OOMs long runs.
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
".venv\Scripts\python.exe" -m src.run %*
