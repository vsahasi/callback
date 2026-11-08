import { handleGet } from "@/lib/handlers";

export const dynamic = "force-dynamic";

export const GET = async (_req: Request, ctx: { params: Promise<{ id: string }> }) =>
  handleGet((await ctx.params).id);
