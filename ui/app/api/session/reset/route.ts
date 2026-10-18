import { errorResponse, HttpError } from "@/lib/data";
import { resetSession } from "@/lib/engine";

// Oturum sıfırlama ({"session"}) → motorun /v1/session/reset'i.
export async function POST(req: Request) {
  try {
    let body: unknown;
    try {
      body = await req.json();
    } catch {
      throw new HttpError(400, "geçersiz JSON gövdesi");
    }
    const session = (body as { session?: unknown })?.session;
    if (typeof session !== "string" || session === "") throw new HttpError(400, "session zorunlu");
    return await resetSession(session);
  } catch (e) {
    return errorResponse(e);
  }
}
