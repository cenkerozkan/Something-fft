"""
ScriptRunner
------------

GUI’den seçilen parametrelerle (CSV yolu, örnekleme frekansı, çıktı klasörü)
komut-satırı için yazılmış FFT analiz betiklerini sorunsuz çalıştırır.

Kullanım:
    from gui.script_runner import ScriptRunner

    runner = ScriptRunner("fft_very_first_with_all_things")
    log_txt = runner.run(
        csv_path     = Path("/tam/yol/dosya.csv"),
        sample_rate  = 2600.0,
        out_dir      = Path("/nereye/yazacak")
    )
    print(log_txt)
"""

from __future__ import annotations

import importlib
import os
import sys
import io
import shutil
from pathlib import Path


class ScriptRunner:
    """Bir analiz betiğini GUI’den parametre aktararak çalıştırır."""

    # Betiklerin içinde sabit olarak beklenen dosya adı
    _EXPECTED_CSV_NAME = "updated_vibration_data.csv"

    def __init__(self, module_name: str) -> None:
        """
        Parameters
        ----------
        module_name : str
            Örneğin "fft_very_first_with_all_things_resultant"
        """
        self.module_name = module_name
        self.mod = importlib.import_module(module_name)

    # ------------------------------------------------------------------ #
    def run(self, csv_path: Path, sample_rate: float, out_dir: Path) -> str:
        """
        Betiği çalıştır ve ürettiği stdout/std­err’i metin olarak geri döndür.

        Parameters
        ----------
        csv_path : Path
            Kullanıcının seçtiği ham veri dosyası.
        sample_rate : float
            Örnekleme frekansı (Hz). Betiğin input() çağrılarına enjekte edilir.
        out_dir : Path
            Betiğin OUTPUT_DIR sabitinin üzerine yazılır.

        Returns
        -------
        str
            Betikten yakalanan terminal çıktısı.
        """
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)

        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) Betikteki global sabitleri/yöntemleri yamala
        self.mod.OUTPUT_DIR = str(out_dir)           # Çıktı klasörü
        self.mod.input      = lambda _: str(sample_rate)  # input() -> örnekleme

        # 2) Çalışma dizinini geçici olarak CSV’nin olduğu klasöre taşı
        cwd_before = Path.cwd()
        temp_copy  = None
        stdout_buf = io.StringIO()

        try:
            os.chdir(csv_path.parent)

            # 3) Betik sabit ismi arıyorsa kopya oluştur
            if csv_path.name != self._EXPECTED_CSV_NAME:
                temp_copy = Path(self._EXPECTED_CSV_NAME)
                if temp_copy.exists():
                    temp_copy.unlink()
                shutil.copy(csv_path.name, temp_copy)

            # 4) stdout/stderr’i yakala
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = stdout_buf
            try:
                # <<<  BU İKİ SATIRI EKLEYİN  >>>
                import matplotlib
                matplotlib.use("Agg")  # GUI’siz backend
                import matplotlib.pyplot as plt
                plt.show = lambda *a, **k: None  # show() → hiçbir şey yapma
                # <<<  EKLEME BİTTİ            >>>

                self.mod.main()  # Betiğin orijinal akışı
            finally:
                sys.stdout, sys.stderr = old_out, old_err

        finally:
            # 5) Temizlik
            if temp_copy and temp_copy.exists():
                temp_copy.unlink()
            os.chdir(cwd_before)

        return stdout_buf.getvalue()
