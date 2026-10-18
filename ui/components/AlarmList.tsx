// Açık alarmlar ve oturum boyunca başlama / temizlenme geçmişi (Faz 3 tasarımı §4.4).
import { ALARM_TEXT, type AlarmEvent, type AlarmId } from "@/lib/alarms";

export function AlarmList({ active, history }: { active: AlarmId[]; history: AlarmEvent[] }) {
  const recent = history.slice(-20).reverse();
  return (
    <section className="flex flex-col gap-2 rounded border border-line bg-panel p-3">
      <h2 className="text-sm font-semibold">Alarmlar</h2>
      {active.length === 0 ? (
        <p className="text-sm text-muted">Açık alarm yok.</p>
      ) : (
        <ul className="flex flex-col gap-1" role="status">
          {active.map((id) => (
            <li
              key={id}
              className="rounded border border-alarm/70 bg-alarm/15 px-2 py-1 text-sm font-medium"
            >
              {ALARM_TEXT[id]}
            </li>
          ))}
        </ul>
      )}
      {recent.length > 0 && (
        <ol className="max-h-32 overflow-y-auto text-xs" aria-label="Alarm geçmişi">
          {recent.map((e, i) => (
            <li key={`${e.frame}-${e.id}-${e.kind}-${i}`} className="flex gap-2 py-0.5">
              <span className="num w-14 text-muted">kare {e.frame}</span>
              <span>
                {ALARM_TEXT[e.id]} {e.kind === "start" ? "başladı" : "temizlendi"}
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
