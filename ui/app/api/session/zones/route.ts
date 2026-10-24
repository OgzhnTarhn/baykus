// Bölgeler (Faz 4 §5.5, §5.8): tarayıcı motoru görmez; gövde doğrulanmadan motora iletilir.
import { errorResponse, HttpError } from "@/lib/data";
import { getZones, putZones } from "@/lib/engine";

const MAX_BODY = 64 * 1024;

function sessionOf(req: Request): string {
  const session = new URL(req.url).searchParams.get("session");
  if (session === null || session === "") throw new HttpError(400, "session zorunlu");
  return session;
}

export async function GET(req: Request) {
  try {
    return await getZones(sessionOf(req));
  } catch (e) {
    return errorResponse(e);
  }
}

export async function PUT(req: Request) {
  try {
    const session = sessionOf(req);
    const text = await req.text();
    if (new TextEncoder().encode(text).length > MAX_BODY) {
      throw new HttpError(413, "gövde 64 KB'tan büyük");
    }
    return await putZones(session, text);
  } catch (e) {
    return errorResponse(e);
  }
}
