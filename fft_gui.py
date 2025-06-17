"""
GPT GENERATED CODE!
"""

from __future__ import annotations
import sys, datetime, zoneinfo
from pathlib import Path
from PySide6.QtCore    import Qt, QThread, Signal
from PySide6.QtGui     import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QFormLayout
)
from script_runner import ScriptRunner

# Çalıştırılacak betikler (sırası önemli değil)
MODULE_LIST = [
    "fft_very_first_with_all_things_resultant",
    "fft_very_first_with_all_things",
    "fft_very_first_with_distribution",
]

def default_output_folder() -> Path:
    ts = datetime.datetime.now(zoneinfo.ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d_%H-%M-%S")
    p = Path.cwd() / f"fft_results_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p

# --------------------------------------------------------------------
class Worker(QThread):
    log      = Signal(str)
    finished = Signal()

    def __init__(self, runner: ScriptRunner, csv: Path, rate: float, out_dir: Path):
        super().__init__()
        self.runner, self.csv, self.rate, self.out_dir = runner, csv, rate, out_dir

    def run(self):
        try:
            text = self.runner.run(self.csv, self.rate, self.out_dir)
            self.log.emit(text)
        except Exception as exc:
            self.log.emit(f"[HATA] {exc}\n")
        self.finished.emit()

# --------------------------------------------------------------------
class FFTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration FFT Analyzer")
        self._build_ui()
        self.workers: list[Worker] = []

    # ---------------------- UI --------------------------------------
    def _build_ui(self):
        self.csv_edit  = QLineEdit(placeholderText="CSV dosyası seçin…")
        self.csv_btn   = QPushButton("Gözat")
        self.rate_edit = QLineEdit(placeholderText="2600.0")
        self.out_edit  = QLineEdit(str(default_output_folder()))
        self.out_btn   = QPushButton("Gözat")
        self.run_btn   = QPushButton("Analizi Başlat")
        self.log_box   = QPlainTextEdit(readOnly=True)

        form = QFormLayout()
        h1 = QHBoxLayout(); h1.addWidget(self.csv_edit); h1.addWidget(self.csv_btn)
        h2 = QHBoxLayout(); h2.addWidget(self.out_edit); h2.addWidget(self.out_btn)
        form.addRow("CSV dosyası:",            h1)
        form.addRow("Örnekleme frekansı (Hz):", self.rate_edit)
        form.addRow("Çıktı klasörü:",          h2)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addWidget(self.run_btn, alignment=Qt.AlignHCenter)
        v.addWidget(QLabel("Log:"))
        v.addWidget(self.log_box)

        container = QWidget(); container.setLayout(v)
        self.setCentralWidget(container)

        self.csv_btn.clicked.connect(self._pick_csv)
        self.out_btn.clicked.connect(self._pick_out_dir)
        self.run_btn.clicked.connect(self._start_analysis)

    # ---------------------- Kullanıcı etkileşimi --------------------
    def _pick_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "CSV seç", "", "CSV Files (*.csv)")
        if file: self.csv_edit.setText(file)

    def _pick_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı klasörü seç")
        if folder: self.out_edit.setText(folder)

    def _start_analysis(self):
        csv_path = Path(self.csv_edit.text())
        if not csv_path.is_file():
            QMessageBox.warning(self, "Eksik CSV", "Geçerli bir CSV dosyası seçin."); return
        try:
            rate = float(self.rate_edit.text() or "0")
            if rate <= 0: raise ValueError
        except ValueError:
            QMessageBox.warning(self, "Hatalı frekans", "Pozitif bir sayı girin."); return

        out_root = Path(self.out_edit.text() or default_output_folder())
        out_root.mkdir(parents=True, exist_ok=True)

        self.run_btn.setEnabled(False); self.log_box.clear()
        self._launch_workers(csv_path, rate, out_root)

    # ---------------------- Worker zinciri --------------------------
    def _launch_workers(self, csv: Path, rate: float, out_root: Path):
        self.workers.clear()

        for mod_name in MODULE_LIST:
            runner = ScriptRunner(mod_name)
            subdir = out_root / Path(runner.mod.OUTPUT_DIR).name  # betiğin orijinal ismi
            subdir.mkdir(parents=True, exist_ok=True)

            w = Worker(runner, csv, rate, subdir)
            w.log.connect(self._append_log)
            self.workers.append(w)

        # zincir şeklinde: biri bitince diğeri başlar
        for i in range(len(self.workers) - 1):
            self.workers[i].finished.connect(self.workers[i+1].start)
        self.workers[-1].finished.connect(self._analysis_done)

        self.workers[0].start()

    def _append_log(self, text: str):
        self.log_box.moveCursor(QTextCursor.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QTextCursor.End)

    def _analysis_done(self):
        self.run_btn.setEnabled(True)
        QMessageBox.information(self, "Tamamlandı", "Tüm FFT analizleri bitti!")

# --------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    win = FFTMainWindow(); win.resize(780, 560); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
