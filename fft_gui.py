# gui/fft_gui.py
from __future__ import annotations

import sys, datetime, zoneinfo, os
from pathlib import Path

from PySide6.QtCore    import Qt, QThread, Signal
from PySide6.QtGui     import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QComboBox, QPlainTextEdit, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QFormLayout
)

from script_runner import ScriptRunner


# ──────────────────────────────────────────────────────────────
def default_output_folder() -> Path:
    """Türkiye saatiyle timestamp’li klasör üretir."""
    ts = datetime.datetime.now(
        zoneinfo.ZoneInfo("Europe/Istanbul")
    ).strftime("%Y-%m-%d_%H-%M-%S")
    p = Path.cwd() / f"fft_results_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────────────────────────────────────────────────
class Worker(QThread):
    """Analizi arka planda çalıştırır, GUI’yi kilitlemez."""
    log      = Signal(str)
    finished = Signal()

    def __init__(self, runner: ScriptRunner,
                 csv_path: Path, rate: float, out_dir: Path):
        super().__init__()
        self.runner   = runner
        self.csv_path = csv_path
        self.rate     = rate
        self.out_dir  = out_dir

    def run(self):
        try:
            text = self.runner.run(
                csv_path    = self.csv_path,
                sample_rate = self.rate,
                out_dir     = self.out_dir
            )
            self.log.emit(text)
        except Exception as exc:
            self.log.emit(f"[HATA] {exc}\n")
        self.finished.emit()


# ──────────────────────────────────────────────────────────────
class FFTMainWindow(QMainWindow):
    """Ana pencere."""

    _MODULES = {
        "Resultant (tek eksen)"        : "fft_very_first_with_all_things_resultant",
        "XYZ + Resultant (tam)"        : "fft_very_first_with_all_things",
        "XYZ + Polar distribution"     : "fft_very_first_with_distribution",
    }

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration FFT Analyzer")
        self._build_ui()
        self.worker: Worker | None = None

    # ----------------------------------------------------------
    def _build_ui(self):
        # --- Girdi alanları
        self.csv_edit  = QLineEdit(placeholderText="CSV dosyası seçin…")
        self.csv_btn   = QPushButton("Gözat")
        self.rate_edit = QLineEdit(placeholderText="2600.0")
        self.out_edit  = QLineEdit(str(default_output_folder()))
        self.out_btn   = QPushButton("Gözat")
        self.mode_box  = QComboBox()
        self.mode_box.addItems(self._MODULES.keys())
        self.run_btn   = QPushButton("Analizi Başlat")
        self.log_box   = QPlainTextEdit(readOnly=True)

        # --- Form yerleşimi
        form = QFormLayout()
        h1 = QHBoxLayout(); h1.addWidget(self.csv_edit); h1.addWidget(self.csv_btn)
        h2 = QHBoxLayout(); h2.addWidget(self.out_edit); h2.addWidget(self.out_btn)
        form.addRow("CSV dosyası:",          h1)
        form.addRow("Örnekleme frekansı (Hz):", self.rate_edit)
        form.addRow("Çıktı klasörü:",        h2)
        form.addRow("Analiz türü:",          self.mode_box)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addWidget(self.run_btn, alignment=Qt.AlignHCenter)
        v.addWidget(QLabel("Log:"))
        v.addWidget(self.log_box)

        container = QWidget(); container.setLayout(v)
        self.setCentralWidget(container)

        # --- Sinyaller
        self.csv_btn.clicked.connect(self._pick_csv)
        self.out_btn.clicked.connect(self._pick_out_dir)
        self.run_btn.clicked.connect(self._start_analysis)

    # ----------------------------------------------------------
    def _pick_csv(self):
        file, _ = QFileDialog.getOpenFileName(
            self, "CSV seç", "", "CSV Files (*.csv)")
        if file:
            self.csv_edit.setText(file)

    def _pick_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı klasörü seç")
        if folder:
            self.out_edit.setText(folder)

    # ----------------------------------------------------------
    def _start_analysis(self):
        csv_path = Path(self.csv_edit.text())
        if not csv_path.is_file():
            QMessageBox.warning(self, "Eksik CSV", "Geçerli bir CSV dosyası seçin.")
            return
        try:
            rate = float(self.rate_edit.text() or "0")
            if rate <= 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Hatalı frekans", "Pozitif bir sayı girin.")
            return

        out_dir = Path(self.out_edit.text() or default_output_folder())
        out_dir.mkdir(parents=True, exist_ok=True)

        module_name = self._MODULES[self.mode_box.currentText()]
        runner = ScriptRunner(module_name)

        self.run_btn.setEnabled(False)
        self.log_box.clear()

        self.worker = Worker(runner, csv_path, rate, out_dir)
        self.worker.log.connect(self._append_log)
        self.worker.finished.connect(self._analysis_done)
        self.worker.start()

    # ----------------------------------------------------------
    def _append_log(self, text: str):
        self.log_box.moveCursor(QTextCursor.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QTextCursor.End)

    def _analysis_done(self):
        self.run_btn.setEnabled(True)
        QMessageBox.information(self, "Tamamlandı",
                                "FFT analizi bitti!")

def main():
    app = QApplication(sys.argv)
    win = FFTMainWindow(); win.resize(750, 550); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()