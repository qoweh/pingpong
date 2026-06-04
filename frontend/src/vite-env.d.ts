/// <reference types="vite/client" />

declare module "@mujoco/mujoco/mujoco.wasm?url" {
  const src: string;
  export default src;
}
