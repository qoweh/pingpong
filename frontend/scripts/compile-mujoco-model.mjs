import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import loadMujoco from "@mujoco/mujoco";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendRoot = path.resolve(__dirname, "..");
const assetRoot = path.join(frontendRoot, "public", "assets", "mujoco");
const manifestPath = path.join(assetRoot, "asset-manifest.json");
const outputScene = "pingpong_scene.mjb";

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
const sourceScene = manifest.sourceScene ?? (manifest.sceneFormat === "mjb" ? "scene.xml" : manifest.scene);
const sourceFiles = manifest.sourceFiles ?? manifest.files;

const wasmPath = path.join(frontendRoot, "node_modules", "@mujoco", "mujoco", "mujoco.wasm");
const mujoco = await loadMujoco({
  locateFile: (file) => (file.endsWith(".wasm") ? wasmPath : file)
});

const vfs = new mujoco.MjVFS();
for (const file of sourceFiles) {
  vfs.addBuffer(file, fs.readFileSync(path.join(assetRoot, file)));
}

const model = mujoco.MjModel.from_xml_path(sourceScene, vfs);
mujoco.mj_saveModel(model, outputScene, null);
const mjbBytes = mujoco.FS.readFile(outputScene);
fs.writeFileSync(path.join(assetRoot, outputScene), mjbBytes);

const nextManifest = {
  ...manifest,
  scene: outputScene,
  sceneFormat: "mjb",
  files: [outputScene],
  sourceScene,
  sourceFiles
};
fs.writeFileSync(manifestPath, `${JSON.stringify(nextManifest, null, 2)}\n`);

console.log(`Compiled ${sourceScene} to ${outputScene} (${mjbBytes.length.toLocaleString()} bytes).`);

model.delete();
vfs.delete();
