import { NextResponse, type NextRequest } from "next/server";
import { checkAuth } from "@/lib/auth";

export async function middleware(req: NextRequest) {
  const verdict = await checkAuth(req.headers.get("authorization"), process.env);
  if (verdict === "ok") return NextResponse.next();
  if (verdict === "unconfigured")
    return new NextResponse(
      "Dashboard is not configured: set DASHBOARD_PASSWORD (and DASHBOARD_USER). It will not serve without it.",
      { status: 503, headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" } },
    );
  return new NextResponse("Authentication required.", {
    status: 401,
    headers: {
      "WWW-Authenticate": 'Basic realm="Callback dashboard", charset="UTF-8"',
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
    },
  });
}

// Every route: pages, /api and static assets.
export const config = { matcher: "/:path*" };
