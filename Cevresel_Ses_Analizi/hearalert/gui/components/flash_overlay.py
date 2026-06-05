"""
HearAlert - Flash Overlay (Gorsel Uyari)
===========================================
Guvenlik-kritik ses algilandiginda ekran kirmizi yanip soner.
Normal sesler icin kisa sari flas.
ctk.after() ile non-blocking animasyon.
"""

import customtkinter as ctk
from gui.styles import theme


class FlashOverlay(ctk.CTkFrame):
    """Tam ekran flas uyari bileeni.

    Guvenlik-kritik sesler icin 5x kirmizi flas,
    normal sesler icin 1x sari flas.
    """

    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        self.parent = parent
        self._flash_jobs = []
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lower()  # Basta gorunmez

    def trigger_flash(self, is_critical: bool = False):
        """Flas animasyonunu baslatir."""
        self.cancel()
        if is_critical:
            self._flash_on(theme.FLASH_COUNT, True)

    def _flash_on(self, count: int, is_critical: bool):
        """Flas ON (kirmizi)."""
        if count <= 0:
            self._reset()
            return
        self.configure(fg_color=theme.CRITICAL_RED)
        self.lift()
        job = self.after(theme.FLASH_DURATION_MS,
                         lambda: self._flash_off(count - 1, is_critical))
        self._flash_jobs.append(job)

    def _flash_off(self, count: int, is_critical: bool):
        """Flas OFF (seffaf)."""
        self.configure(fg_color="transparent")
        self.lower()
        job = self.after(theme.FLASH_DURATION_MS,
                         lambda: self._flash_on(count, is_critical))
        self._flash_jobs.append(job)

    def _reset(self):
        """Overlay'i sifirla."""
        self.configure(fg_color="transparent")
        self.lower()

    def cancel(self):
        """Devam eden animasyonlari iptal et."""
        for job in self._flash_jobs:
            try:
                self.after_cancel(job)
            except Exception:
                pass
        self._flash_jobs.clear()
        self._reset()
