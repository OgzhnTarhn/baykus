// Atıf (Faz 3 tasarımı §4.4): M3OT ve lisans. Bağlantı yalnız metin; sayfa dış adrese istek yapmaz
// (kapalı ağ kanıtı 3).
export function Footer() {
  return (
    <footer className="border-t border-line px-4 py-2 text-xs text-muted">
      Görüntüler: M3OT veri seti (Scientific Data, 2025), CC BY 4.0 lisansıyla
      (creativecommons.org/licenses/by/4.0). Kutular ve skorlar Baykuş modelinin çıktısıdır, veri
      setinin etiketleri değildir.
    </footer>
  );
}
