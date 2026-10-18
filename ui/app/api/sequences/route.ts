import { errorResponse, listSequences } from "@/lib/data";

// Dizi listesi (test bölümü hariç).
export async function GET() {
  try {
    return Response.json(await listSequences());
  } catch (e) {
    return errorResponse(e);
  }
}
