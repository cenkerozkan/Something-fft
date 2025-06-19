"""
fft_gui.py – Encoder eksikse yapay Encoder oluşturma desteği ve
ardından üç ayrı FFT analiz betiğini ard-arda çalıştıran arayüz.
"""

from __future__ import annotations
import sys, datetime, zoneinfo, subprocess, csv
from pathlib import Path

from PySide6.QtCore    import Qt, QThread, Signal
from PySide6.QtGui     import QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton,
    QPlainTextEdit, QFileDialog, QMessageBox,
    QHBoxLayout, QVBoxLayout, QFormLayout
)

from script_runner import ScriptRunner   # aynı klasördeki yardımcı

# Çalıştırılacak betikler
MODULE_LIST = [
    "fft_very_first_with_all_things_resultant",
    "fft_very_first_with_all_things",
    "fft_very_first_with_distribution",
]

# ─────────────────────────────────────────────────────────────
def default_output_folder() -> Path:
    ts = datetime.datetime.now(zoneinfo.ZoneInfo("Europe/Istanbul")).strftime("%Y-%m-%d_%H-%M-%S")
    p = Path.cwd() / f"fft_results_{ts}"
    p.mkdir(parents=True, exist_ok=True)
    return p

# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
class FFTMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vibration FFT Analyzer")
        self._build_ui()
        self.workers: list[Worker] = []

    # ---------------- UI kurulumu ----------------
    def _build_ui(self):
        self.csv_edit  = QLineEdit(placeholderText="CSV dosyası seçin…")
        self.csv_btn   = QPushButton("Gözat")
        self.rate_edit = QLineEdit(placeholderText="2600.0")
        self.out_edit  = QLineEdit(str(default_output_folder()))
        self.out_btn   = QPushButton("Gözat")
        self.enc_btn   = QPushButton("Encoder Oluştur")   # yeni buton
        self.enc_btn.setEnabled(False)                    # varsayılan kapalı
        self.run_btn   = QPushButton("Analizi Başlat")
        self.log_box   = QPlainTextEdit(readOnly=True)

        # Form yerleşimi
        form = QFormLayout()
        row_csv = QHBoxLayout(); row_csv.addWidget(self.csv_edit); row_csv.addWidget(self.csv_btn)
        row_out = QHBoxLayout(); row_out.addWidget(self.out_edit); row_out.addWidget(self.out_btn)
        form.addRow("CSV dosyası:",            row_csv)
        form.addRow("Örnekleme frekansı (Hz):", self.rate_edit)
        form.addRow("Çıktı klasörü:",          row_out)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addWidget(self.enc_btn, alignment=Qt.AlignHCenter)
        v.addWidget(self.run_btn, alignment=Qt.AlignHCenter)
        v.addWidget(QLabel("Log:"))
        v.addWidget(self.log_box)

        container = QWidget(); container.setLayout(v)
        self.setCentralWidget(container)

        # Sinyaller
        self.csv_btn.clicked.connect(self._pick_csv)
        self.out_btn.clicked.connect(self._pick_out_dir)
        self.enc_btn.clicked.connect(self._run_encoder_loader)
        self.run_btn.clicked.connect(self._start_analysis)

    # ---------------- CSV seçimi ----------------
    def _pick_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "CSV seç", "", "CSV Files (*.csv)")
        if file:
            self.csv_edit.setText(file)
            self._check_encoder_status()

    def _check_encoder_status(self):
        """CSV'de Encoder sütunu yoksa veya tüm değerler 0 ise Encoder Oluştur butonunu aç."""
        needs = False
        csv_path = Path(self.csv_edit.text())
        if csv_path.is_file():
            try:
                with open(csv_path, newline='') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        needs = True
                    elif 'Encoder' not in reader.fieldnames:
                        needs = True
                    else:
                        # İlk 100 satırda sıfırdan farklı bir değer var mı?
                        zero_only = True
                        for _, row in zip(range(100), reader):
                            try:
                                if float(row['Encoder']) != 0.0:
                                    zero_only = False
                                    break
                            except:
                                pass
                        needs = zero_only
            except Exception:
                needs = False
        self.enc_btn.setEnabled(needs)

    # ---------------- Encoder üretimi ----------------
    def _run_encoder_loader(self):
        csv_path = Path(self.csv_edit.text())
        if not csv_path.is_file():
            QMessageBox.warning(self, "Dosya Yok", "Önce geçerli bir CSV seçin.")
            return

        try:
            # encoder_loader.py, bu GUI dosyası ile aynı klasörde
            enc_path = Path(__file__).with_name("encoder_loader.py")

            proc = subprocess.run(
                [sys.executable, str(enc_path)],
                input=str(csv_path.name) + "\n",      # stdin: dosya adı
                text=True,
                capture_output=True,
                cwd=csv_path.parent,                  # çıktı CSV’nin yanına
                timeout=120
            )

            self.log_box.appendPlainText(proc.stdout)

            if proc.returncode == 0:
                QMessageBox.information(self, "Tamam",
                                        "updated_vibration_data.csv oluşturuldu.")
            else:
                QMessageBox.warning(self, "Hata",
                                    f"encoder_loader.py çalışırken hata:\n{proc.stderr}")
        except Exception as e:
            QMessageBox.warning(self, "Hata",
                                f"encoder_loader.py başlatılamadı:\n{e}")

        # Oluşturma sonrası tekrar kontrol et
        self._check_encoder_status()

    # ---------------- Çıktı klasörü seçimi ---------------
    def _pick_out_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Çıktı klasörü seç")
        if folder:
            self.out_edit.setText(folder)

    # ---------------- FFT analiz akışı -------------------
    def _start_analysis(self):
        csv_path = Path(self.csv_edit.text())
        if not csv_path.is_file():
            QMessageBox.warning(self, "Eksik CSV", "Geçerli bir CSV dosyası seçin.")
            return
        try:
            rate = float(self.rate_edit.text() or "0")
            assert rate > 0
        except (ValueError, AssertionError):
            QMessageBox.warning(self, "Hatalı frekans", "Pozitif bir sayı girin.")
            return

        out_root = Path(self.out_edit.text() or default_output_folder())
        out_root.mkdir(parents=True, exist_ok=True)

        self.run_btn.setEnabled(False)
        self.log_box.clear()
        self.workers.clear()

        # Worker zinciri
        for mod in MODULE_LIST:
            runner = ScriptRunner(mod)
            subdir = out_root / Path(runner.mod.OUTPUT_DIR).name
            subdir.mkdir(parents=True, exist_ok=True)

            w = Worker(runner, csv_path, rate, subdir)
            w.log.connect(self._append_log)
            self.workers.append(w)

        for i in range(len(self.workers) - 1):
            self.workers[i].finished.connect(self.workers[i + 1].start)
        self.workers[-1].finished.connect(self._analysis_done)

        self.workers[0].start()

    def _append_log(self, text: str):
        self.log_box.moveCursor(QTextCursor.End)
        self.log_box.insertPlainText(text)
        self.log_box.moveCursor(QTextCursor.End)

    def _analysis_done(self):
        self.run_btn.setEnabled(True)
        QMessageBox.information(self, "Tamamlandı", "Tüm FFT analizleri bitti!")

# ─────────────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    win = FFTMainWindow()
    win.resize(800, 580)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
