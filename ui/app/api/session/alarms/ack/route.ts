import { errorResponse, HttpError } from "@/lib/data";
import { ackAlarm } from "@/lib/engine";

// Alarm onaylama ({"session", "alarm"}) → motorun /v1/session/alarms/ack'i.
export async function POST(req: Request) {
  try {
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      throw new HttpError(400, "geçersiz JSON gövdesi");
    }
    const { session, alarm } = (body ?? {}) as { session?: unknown; alarm?: unknown };
    if (typeof session !== "string" || session === "" || typeof alarm !== "string" || alarm === "") {
      throw new HttpError(400, "session ve alarm zorunlu");
    }
    return await ackAlarm(session, alarm);
  } catch (e) {
    return errorResponse(e);
  }
}
