import type { ResetScope } from "@/lib/client";
import { errorResponse, HttpError } from "@/lib/data";
import { resetSession } from "@/lib/engine";

// Oturum sıfırlama ({"session", "scope"?}) → motorun /v1/session/reset'i.
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
    const raw = (body as { scope?: unknown })?.scope;
    if (raw !== undefined && raw !== "all" && raw !== "tracks") {
      throw new HttpError(400, `bilinmeyen scope: ${String(raw)}`);
    }
    const scope = raw as ResetScope | undefined;
    return await resetSession(session, scope);
  } catch (e) {
    return errorResponse(e);
  }
}
