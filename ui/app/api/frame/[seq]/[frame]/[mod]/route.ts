import { errorResponse, parseModality, readFrame } from "@/lib/data";

// Tek karenin özgün baytları (RGB JPEG, IR PNG); yol bileşenleri doğrulanır (400 / 404).
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ seq: string; frame: string; mod: string }> },
) {
  try {
    const { seq, frame, mod } = await params;
    const m = parseModality(mod);
    const bytes = await readFrame(seq, frame, m);
    return new Response(new Uint8Array(bytes), {
      headers: {
        "content-type": m === "rgb" ? "image/jpeg" : "image/png",
        "cache-control": "private, max-age=3600",
      },
    });
  } catch (e) {
    return errorResponse(e);
  }
}
