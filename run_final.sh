set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/Scripts/python.exe
EXPF=configs/experiments_final.yaml
M_UNETPP=sweep_unetpp_dbz0_e012345678_focaltv
M_ATTUNET=sweep_attunet_dbz0_e012345678_focaltv
echo "===== [1/3] single-model TTA re-eval (both 9-elev members) ====="
$PY -m src.evaluate --experiments $EXPF --split test --tta
echo "===== [2/3] ensemble, NO TTA ====="
$PY -m src.evaluate --experiments $EXPF --ensemble --split test --names "$M_ATTUNET,$M_UNETPP"
cp outputs/ensemble_test.json outputs/ensemble_test_nott.json
echo "===== [3/3] ensemble, WITH TTA ====="
$PY -m src.evaluate --experiments $EXPF --ensemble --split test --tta --names "$M_ATTUNET,$M_UNETPP"
cp outputs/ensemble_test.json outputs/ensemble_test_tta.json
echo "===== DONE ====="
