import { errorResponse, HttpError } from "@/lib/data";
import { getTrack } from "@/lib/engine";

// İz geçmişi (?session&id) → motorun /v1/session/track'i.
export async function GET(req: Request) {
  try {
    const q = new URL(req.url).searchParams;
    const session = q.get("session");
    if (session === null || session === "") throw new HttpError(400, "session zorunlu");
    const id = q.get("id");
    if (id === null || !/^\d+$/.test(id)) throw new HttpError(400, "id tam sayı olmalı");
    return await getTrack(session, Number(id));
  } catch (e) {
    return errorResponse(e);
  }
}
