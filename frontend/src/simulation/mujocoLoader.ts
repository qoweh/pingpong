import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import type { MainModule } from "@mujoco/mujoco";

let cachedModule: Promise<MainModule> | null = null;

export function loadMujocoModule(): Promise<MainModule> {
  if (!cachedModule) {
    cachedModule = loadMujoco({
      locateFile: (path: string) => (path.endsWith(".wasm") ? mujocoWasmUrl : path)
    });
  }

  return cachedModule;
}
