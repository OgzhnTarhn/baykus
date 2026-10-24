import { errorResponse, HttpError } from "@/lib/data";
import { getEvents } from "@/lib/engine";

// Oturum olayları (?session&since&format) → motorun /v1/session/events'i; yanıt olduğu gibi iletilir.
export async function GET(req: Request) {
  try {
    const q = new URL(req.url).searchParams;
    const session = q.get("session");
    if (session === null || session === "") throw new HttpError(400, "session zorunlu");
    const since = q.get("since") ?? "0";
    if (!/^\d+$/.test(since)) throw new HttpError(400, "since 0 ya da daha büyük tam sayı olmalı");
    const format = q.get("format") ?? "json";
    if (format !== "json" && format !== "csv") throw new HttpError(400, `bilinmeyen format: ${format}`);
    return await getEvents(session, Number(since), format);
  } catch (e) {
    return errorResponse(e);
  }
}
