"""
script_runner.py
GUI'den gelen parametrelerle (csv, sample_rate, out_dir) FFT betiklerini
arka planda çalıştırır; stdout+stderr'i metin olarak döndürür.
"""

from __future__ import annotations
import importlib, os, sys, io, shutil
from pathlib import Path

class ScriptRunner:
    _EXPECTED_CSV_NAME = "updated_vibration_data.csv"

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name
        self.mod = importlib.import_module(module_name)

    # --------------------------------------------------------------
    def run(self, csv_path: Path, sample_rate: float, out_dir: Path) -> str:
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Betik içi sabitleri yamala
        self.mod.OUTPUT_DIR = str(out_dir)          # nereye yazacağını söyle
        self.mod.input      = lambda _: str(sample_rate)  # input() → sample_rate

        # stdout yakalayacağız
        stdout_buf = io.StringIO()
        cwd_before, temp_copy = Path.cwd(), None

        try:
            os.chdir(csv_path.parent)

            # Beklenen isim farklıysa geçici kopya oluştur
            if csv_path.name != self._EXPECTED_CSV_NAME:
                temp_copy = Path(self._EXPECTED_CSV_NAME)
                if temp_copy.exists():
                    temp_copy.unlink()
                shutil.copy(csv_path.name, temp_copy)

            # Matplotlib'i penceresiz moda al
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.show = lambda *a, **k: None

            # stdout/stderr yönlendir
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = stdout_buf
            try:
                self.mod.main()
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        finally:
            if temp_copy and temp_copy.exists():
                temp_copy.unlink()
            os.chdir(cwd_before)

        stdout_buf.write(f"\n--- {self.module_name} TAMAMLANDI ---\n")
        return stdout_buf.getvalue()
