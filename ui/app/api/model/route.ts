// Yüklü model bilgisi → motorun /v1/model'i (Faz 4 §5.7: motion parametreleri ve sha256'ları).
import { errorResponse } from "@/lib/data";
import { getModel } from "@/lib/engine";

export async function GET() {
  try {
    return await getModel();
  } catch (e) {
    return errorResponse(e);
  }
}
